import os
from torch.nn.utils import clip_grad_norm_

os.environ["WANDB_KEY"] = "017a2798d0b6bc80eecaa69d5f85b84cd4ca556f"
os.environ["PROJECT"] = "diff-fal"
os.environ["ENTITY"] = "r21"

from vdit_mflow import SiT_models
from eval import Eval
import click
import torch
import torch.nn.functional as F
import numpy as np
import json
import random
import torch.distributed as dist
from torch.backends.cuda import (
    enable_cudnn_sdp,
    enable_flash_sdp,
    enable_math_sdp,
    enable_mem_efficient_sdp,
)
from torchvision.utils import make_grid
import copy
# sdpa
from torch.nn.attention import SDPBackend, sdpa_kernel

from datetime import datetime
from PIL import Image
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
import wandb
import sys
import logging
from collections import OrderedDict
from tqdm import tqdm, trange
from transformers import AutoModel, AutoImageProcessor
from diffusers import AutoencoderKL
import pickle
##############################################################################
#                          Dataset Definition (IMAGENET)                     #
##############################################################################

class IMAGENET(torch.utils.data.Dataset):
    def __init__(self, is_train=True):
        dpath = "./inet/inet.train.pkl" if is_train else "./inet/inet.val.pkl"
        with open(dpath, "rb") as f:
            self.labels, self.latents = pickle.load(f)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        image = self.latents[idx]
        label, _ = self.labels[idx]
        # Reshape from (32, 32, 4) to (4, 32, 32)
        image = image.astype(np.float32).reshape(32, 32, 4)
        image = np.transpose(image, (2, 0, 1))
        # image = ((image / 255.0) * 2) - 1
        # image = image * 16.0
        return image, int(label)


##############################################################################
#                                Helpers                                     #
##############################################################################

def requires_grad(model, flag=True):
    for p in model.parameters():
        p.requires_grad = flag

def cleanup():
    dist.destroy_process_group()

def create_logger(logging_dir):
    if dist.get_rank() == 0:
        logging.basicConfig(
            level=logging.INFO,
            format='[\033[34m%(asctime)s\033[0m] %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S',
            handlers=[logging.StreamHandler(), logging.FileHandler(f"{logging_dir}/log.txt")]
        )
        logger = logging.getLogger(__name__)
    else:
        logger = logging.getLogger(__name__)
        logger.addHandler(logging.NullHandler())
    return logger

##############################################################################
#                              Stable Diffusion VAE                          #
##############################################################################

def sdvae(type, p="int8", device="cuda:0"):
    """
    Creates an AutoencoderKL from the given stable-diffusion VAE weights,
    with optional postprocessing (int8/fp16/fp32).
    """
    name_by_t = {
        "mse": "stabilityai/sd-vae-ft-mse",
        "ema": "stabilityai/sd-vae-ft-ema",
    }
    vae = AutoencoderKL.from_pretrained(name_by_t[type]).to(device)

    def encode(x):
        with torch.no_grad():
            latent = vae.encode(x).latent_dist.sample()
        if p == "int8":
            latent = latent / 16
            latent = (latent + 1) / 2
            latent = latent.clamp(0, 1) * 255
            latent = latent.type(torch.uint8)
        elif p == "fp16":
            latent = latent.type(torch.float16)
        elif p == "fp32":
            latent = latent.type(torch.float32)
        return latent

    def decode(z):
        if p == "int8":
            z = z.type(torch.float32)
            z = z / 255
            z = (z * 2) - 1
            z = z * 16
        z = z.type(torch.float32)
        with torch.no_grad():
            img = vae.decode(z).sample  # (B,3,H,W) in [-1,1]
        return img

    return encode, decode

##############################################################################
#                           Logit-Normal Timestep                            #
##############################################################################

def lognorm(mu=0, sigma=1, size=None):
    """
    Returns samples from a logit-normal distribution in [0,1].
    """
    samples = np.random.normal(mu, sigma, size)
    samples = 1 / (1 + np.exp(-samples))
    return samples


@torch.no_grad()
def update_ema(ema_model, model, decay=0.9999):
    """
    Step the EMA model towards the current model.
    """
    ema_params = OrderedDict(ema_model.named_parameters())
    model_params = OrderedDict(model.named_parameters())

    for name, param in model_params.items():
        # TODO: Consider applying only to params that require_grad to avoid small numerical changes of pos_embed
        ema_params[name].mul_(decay).add_(param.data, alpha=1 - decay)

##############################################################################
#                                 Main Training Loop                          #
##############################################################################

@click.command()
@click.option("--run_name", default="run_1", help="Name of the run")
@click.option("--global_batch_size", default=256, help="Global batch size across all GPUs")
@click.option("--global_seed", default=4, help="Global seed")
@click.option("--per_gpu_batch_size", default=32, help="Per GPU batch size")
@click.option("--num_iterations", default=500_000, help="Number of training iterations")
@click.option("--learning_rate", default=1e-4, help="Learning rate")
@click.option("--sample_every", default=10_000, help="Sample frequency")
@click.option("--val_every", default=2_000, help="Validation frequency")
@click.option("--kdd_every", default=2_000, help="KDD evaluation frequency")
@click.option("--save_every", default=2_000, help="Checkpoint save frequency")
@click.option("--init_ckpt", default=None, help="Path to initial checkpoint")
@click.option("--cfg_scale", default=1.5, help="CFG scale during KDD evaluation")
@click.option("--uncond_prob", default=0.1, help="Probability of dropping label for unconditional training")
@click.option("--optimizer", default="muon", help="Optimizer")
def main(run_name, global_batch_size, global_seed, per_gpu_batch_size, num_iterations,
         learning_rate, sample_every, val_every, kdd_every, save_every, init_ckpt, cfg_scale, uncond_prob, optimizer):

    ddp_rank = int(os.environ["RANK"])
    ddp_local_rank = int(os.environ["LOCAL_RANK"])
    ddp_world_size = int(os.environ["WORLD_SIZE"])
    torch.manual_seed(global_seed + ddp_rank)
    np.random.seed(global_seed + ddp_rank)
    random.seed(global_seed + ddp_rank)

    ##########################################################################
    #                   DDP Initialization and Basic Setup                   #
    ##########################################################################
    val_per_gpu_batch_size = per_gpu_batch_size * 2
    dist.init_process_group(backend="nccl")
    
    device = f"cuda:{ddp_local_rank}"
    torch.cuda.set_device(device)
    master_process = (ddp_rank == 0)


    grad_accum_steps = int(global_batch_size // (per_gpu_batch_size * ddp_world_size))
    date_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_id = f"{date_time}_{run_name}"

    if master_process:
        print(f"Global batch size: {global_batch_size}")
        print(f"Per GPU batch size: {per_gpu_batch_size}")
        print(f"Gradient accumulation steps: {grad_accum_steps}")
        print(f"Effective batch size per step: {per_gpu_batch_size * ddp_world_size}")

        wandb.init(
            project="imagegpt",
            name=run_name,
            config={
                "global_batch_size": global_batch_size,
                "per_gpu_batch_size": per_gpu_batch_size,
                "grad_accum_steps": grad_accum_steps,
                "num_iterations": num_iterations,
                "learning_rate": learning_rate,
                "sample_every": sample_every,
                "val_every": val_every,
                "kdd_every": kdd_every,
                "save_every": save_every,
                "cfg_scale": cfg_scale,
                "uncond_prob": uncond_prob
            },
        )
        wandb.run.log_code(".")

    # Allow tf32 for speed
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True

    model = SiT_models['SiT-B/2']().to(memory_format=torch.channels_last)  # From your code
    
    # ema model
    ema = copy.deepcopy(model)
    # model = torch.compile(model)
    # ema = torch.compile(ema)
    model = model.to(device)
    ema = ema.to(device)    
    requires_grad(ema, False)
    # ema 
    if init_ckpt is not None and master_process:
        print(f"Loading checkpoint from {init_ckpt}")
    if init_ckpt is not None:
        checkpoint = torch.load(init_ckpt, map_location="cpu")
        model.load_state_dict(checkpoint["model"])
        ema.load_state_dict(checkpoint["ema"])

    random_tensor = torch.ones(1000, 1000, device=device) * ddp_rank
    dist.all_reduce(random_tensor, op=dist.ReduceOp.SUM)
    if master_process:
        print(f"Rank {ddp_rank} has value {random_tensor[0, 0].item()}\n")

    # Wrap in DDP
    model = DDP(model, device_ids=[ddp_local_rank], find_unused_parameters=False)

    if optimizer == "adamw":
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=0
        )
    elif optimizer == "muon":
        from muon import Muon
        adamw_params, muon_params = {}, {}
        for name, param in model.named_parameters():
            if param.requires_grad and param.ndim >= 2 and (".bias" not in name):
                muon_params[name] = param
            else:
                adamw_params[name] = param

        for name, param in model.named_parameters():
            if param.requires_grad:
                if (param.ndim >= 2) or ((".bias" not in name) or (name.replace(".bias", ".weight") in muon_params)):
                    # if name has bias, concatenate it with the .weight corresponding to the same layer
                    if ".bias" in name:
                        weight_param = muon_params[name.replace(".bias", ".weight")]
                        bias_param = param
                        muon_params[name.replace(".bias", ".weight")] = [weight_param, bias_param]
                    else:
                        muon_params[name] = param
                else:
                    adamw_params[name] = param
        optimizer = Muon(
            muon_params=list(muon_params.values()),
            lr=learning_rate*4,
            adamw_params=list(adamw_params.values()),
            adamw_lr=learning_rate,
            adamw_wd=0,
            collective_whitening=True
        )

    enable_cudnn_sdp(True)
    enable_flash_sdp(True)
    enable_mem_efficient_sdp(False)
    enable_math_sdp(False)

    ##########################################################################
    #                     Datasets, Samplers, Dataloaders                    #
    ##########################################################################

    train_dataset = IMAGENET(is_train=True)
    val_dataset = IMAGENET(is_train=False)

    train_sampler = torch.utils.data.DistributedSampler(
        train_dataset, num_replicas=ddp_world_size, rank=ddp_rank, shuffle=True, seed=global_seed
    )
    val_sampler = torch.utils.data.DistributedSampler(
        val_dataset, num_replicas=ddp_world_size, rank=ddp_rank, shuffle=False
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=per_gpu_batch_size,
        sampler=train_sampler,
        num_workers=4,
        pin_memory=True,
        drop_last=False,
        persistent_workers=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=val_per_gpu_batch_size,
        sampler=val_sampler,
        num_workers=4,
        pin_memory=True,
        drop_last=False,
    )

    ##########################################################################
    #                         Initialize Evaluator + VAE                     #
    ##########################################################################

    evaluator = Eval()  # Uses DINOv2 for MMD
    encode_fn, decode_fn = sdvae("ema", "fp32", device=device)  # for latents decode

    ##########################################################################
    #                 Prepare fixed class IDs for MMD (KDD) eval             #
    ##########################################################################

    seed_for_rank = global_seed + ddp_rank
    # find appropriate # of samples that's more than 2000 and divisible by world size
    num_kdd_samples = ((2000 // ddp_world_size + 1)) * ddp_world_size
    num_kdd_samples_per_rank = num_kdd_samples // ddp_world_size
    fixed_class_ids = torch.randint(0, 1000, (num_kdd_samples_per_rank,), generator=torch.Generator().manual_seed(seed_for_rank)).to(device)
    ##########################################################################
    #                          Helper Functions                              #
    ##########################################################################

    ctx = torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16)


    # @torch.no_grad()
    # def do_validation():
    #     """
    #     Compute a simple validation loss across the entire val_loader.
    #     """
    #     model.eval()
    #     val_losses = []

    #     for val_latents, val_labels in val_loader:
    #         val_latents = val_latents.to(device, non_blocking=True).to(memory_format=torch.channels_last)
    #         val_labels = val_labels.to(device, non_blocking=True)

    #         # Scale latents by 0.18215 for stable diffusion training
    #         data_val = val_latents * 0.18215

    #         with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
    #             model_kwargs = dict(y=val_labels)
    #             t = torch.rand(data_val.shape[0], device=device)
    #             r = torch.rand(data_val.shape[0], device=device)
                
    #             # Inline rectified flow loss
    #             b = data_val.shape[0]
    #             x_0 = data_val
    #             noise = torch.randn_like(x_0)
    #             x_t = (1 - t[:, None, None, None]) * x_0 + t[:, None, None, None] * noise
    #             # v_pred = model(x_t, t, r, **model_kwargs)
    #             v_target = x_0 - x_t

    #             u, dudt = torch.autograd.functional.jvp(lambda *vs: model(*vs, y=val_labels), (x_t, t, r), (v_target, torch.ones_like(t, requires_grad=True)+0.0, torch.zeros_like(t, requires_grad=True)+0.0))

    #             u_tgt = v_target - (t - r)[:, None, None, None] * dudt
    #             loss = ((u - u_tgt.detach())**2).mean()
                
    #         val_losses.append(loss.item())

    #     val_loss = np.mean(val_losses)
    #     model.train()
    #     return val_loss

    @torch.no_grad()
    def do_ema_sample(num_samples, cfg_scale=1.5):
        """
        Sample images using the EMA model with CFG.
        Returns decoded images in uint8 format [0, 255].
        """
        ema.eval()
        z = torch.randn(num_samples, 4, 32, 32, device=device, 
                        generator=torch.Generator(device=device).manual_seed(seed_for_rank))
        y = fixed_class_ids[:num_samples]

        all_imgs = []

        range_fn_imgs = (lambda *args, **kwargs: trange(*args, **kwargs, position=1)) if master_process else range

        for i in range_fn_imgs(0, num_samples, val_per_gpu_batch_size):
            z_i = z[i:i+val_per_gpu_batch_size]
            y_i = y[i:i+val_per_gpu_batch_size]
            b_i = z_i.size(0)
            
            z_i = torch.cat([z_i, z_i], dim=0).to(memory_format=torch.channels_last)
            ynull = torch.zeros_like(y_i) + 1000
            y_i = torch.cat([y_i, ynull], dim=0)
            model_kwargs = dict(y=y_i)
            
            with sdpa_kernel([SDPBackend.FLASH_ATTENTION, SDPBackend.CUDNN_ATTENTION, 
                             SDPBackend.MATH, SDPBackend.EFFICIENT_ATTENTION]): 
                # Apply classifier-free guidance
                def model_fn_with_cfg(x, t, r, **kwargs):
                    half = x.shape[0] // 2
                    y = kwargs.pop("y")
                    cond_output = ema(x[:half], t[:half], r[:half], y=y[:half])
                    uncond_output = ema(x[half:], t[half:], r[half:], y=y[half:])
                    out =  uncond_output + cfg_scale * (cond_output - uncond_output)
                    return torch.cat([out, out], dim=0)
                
                # Inline Euler sampling
                x = z_i
                t = torch.ones(x.shape[0], device=x.device) 
                r = torch.zeros(x.shape[0], device=x.device) 
                v = model_fn_with_cfg(x, t, r, **model_kwargs)
                x = x - v
                # steps = 50
                # h = 1.0 / steps
                
                # for j in range(steps):
                    
                
                z_i = x
                z_i, _ = z_i.chunk(2, dim=0)  # Remove null class samples  
                z_i = z_i / 0.18215             
                imgs = decode_fn(z_i)
            imgs = (imgs.clamp(-1,1) + 1) * 127.5
            imgs = imgs.type(torch.uint8)
            all_imgs.append(imgs)
        
        return torch.cat(all_imgs, dim=0)

    @torch.no_grad()
    def do_sample_grid(step, cfg_scale=1.5):
        samples = do_ema_sample(per_gpu_batch_size, cfg_scale) # (PGPU, 3, 256, 256)
        all_samples = torch.zeros((global_batch_size, 3, 256, 256), device=device, dtype=samples.dtype)
        dist.all_gather_into_tensor(all_samples, samples)
        # all_samples = all_samples.permute(0, 2, 3, 1)
        x = make_grid(all_samples[:16], nrow=int(np.sqrt(16)))
        x = x.permute(1, 2, 0)
        if master_process:
            sample = Image.fromarray(x.cpu().numpy())
            sample.save(f"sample_{cfg_scale}.jpg", quality=80)
            # sample.save("sample_hq.jpg", quality=95)
            wandb.log({f"samples_{cfg_scale}": wandb.Image(f"./sample_{cfg_scale}.jpg")}, step=step)
        dist.barrier()

    @torch.no_grad()
    def do_kdd_evaluation(cfg_scale):
        imgs = do_ema_sample(num_kdd_samples_per_rank, cfg_scale)
        mmd = evaluator.eval(imgs)
        return mmd

    ##########################################################################
    #                             Training Loop                              #
    ##########################################################################

    update_ema(ema, model.module, 0.0)
    model.train()
    ema.eval()
    train_iter = iter(train_loader)
    optimizer.zero_grad(set_to_none=True)

    pbar = tqdm(range(num_iterations), desc="Training", position=0) if master_process else range(num_iterations)
    running_loss = []
    for step in pbar:
        epoch = step // len(train_loader)
        train_sampler.set_epoch(epoch)
        
        # Gradient Accumulation
        for micro_step in range(grad_accum_steps):
            try:
                latents, labels = next(train_iter)
            except StopIteration:
                train_iter = iter(train_loader)
                latents, labels = next(train_iter)

            latents = latents.to(device, non_blocking=True).to(memory_format=torch.channels_last)
            labels = labels.to(device, non_blocking=True)
            latents = latents * 0.18215

            with ctx:
                model_kwargs = dict(y=labels)
                t = torch.rand(latents.shape[0], device=device)
                r = torch.rand(latents.shape[0], device=device)
                # Inline rectified flow loss
                x_0 = latents
                x_1 = torch.randn_like(x_0)
                x_t = ((1 - t[:, None, None, None]) * x_0) + (t[:, None, None, None] * x_1)
                v_target = x_0 - x_t
                with sdpa_kernel([SDPBackend.MATH]):
                    u, dudt = torch.autograd.functional.jvp(lambda *vs: model(*vs, y=labels), 
                                                            (x_t.detach(), t, r), 
                                                            (v_target.detach() ,
                                                                torch.ones_like(t), 
                                                                torch.zeros_like(t)
                                                            ), create_graph=True)

                u_tgt = v_target - ((t - r)[:, None, None, None] * dudt)
                loss = ((u - u_tgt.detach())**2).mean()

                
                running_loss.append(loss.item())

            loss.backward()
        
        clip_grad_norm_(model.parameters(), max_norm=1.0)  # or any suitable value
        optimizer.step()
        # ema beta should be 0 for first 10k steps and then annealed to 0.999 within 100k steps
        ema_beta = 0.0 if (step < 10000) else 0.999
        update_ema(ema, model.module, ema_beta)
        optimizer.zero_grad(set_to_none=True)

        # Logging
        if master_process and step % 10 == 0:
            wandb.log({
                "train/loss": np.mean(running_loss),
            }, step=step)
            running_loss = []
        # Validation
        # if  step % val_every == 0:
        #     val_loss = do_validation()
        #     if master_process:
        #         wandb.log({"val/loss": val_loss}, step=step)

        # KDD Evaluation
        if  step % kdd_every == 0:
            kdd_10 = do_kdd_evaluation(1.0)
            kdd_15 = do_kdd_evaluation(1.5)
            # kdd_20 = do_kdd_evaluation(2.0)
            # kdd_40 = do_kdd_evaluation(4.0)
            if master_process:
                # print(f"step: {step}, kdd: {kdd:.4f}")
                wandb.log({"kdd/mmd/1.0": kdd_10, "kdd/mmd/1.5": kdd_15})#, "kdd/mmd/2.0": kdd_20, "kdd/mmd/4.0": kdd_40}, step=step)

        # Sample
        if step % sample_every == 0:
            do_sample_grid(step, cfg_scale=1.0)
            do_sample_grid(step, cfg_scale=1.5)
            # do_sample_grid(step, cfg_scale=2.0)
            # do_sample_grid(step, cfg_scale=4.0)

        # Save Checkpoints
        if master_process and step > 0 and step % save_every == 0:
            checkpoint = {
                "model": model.module.state_dict(),
                "optimizer": optimizer.state_dict(),
                "step": step,
                "ema": ema.state_dict(),
            }
            os.makedirs(f"logs/ckpts_{run_id}", exist_ok=True)
            ckpt_path = f"logs/ckpts_{run_id}/step_{step}.pt"
            print(f"Saving checkpoint to {ckpt_path}")
            torch.save(checkpoint, ckpt_path)

    ##########################################################################
    #                           Finishing Up                                 #
    ##########################################################################

    if master_process:
        wandb.finish()
    dist.destroy_process_group()

def test_optimizer():
    # init model
    device = "cuda:0"
    model = SiT_models['SiT-B/2']().to(memory_format=torch.channels_last).to(device)
    # model = torch.compile(model)
    learning_rate = 0.02
    from muon import Muon
    adamw_params, muon_params = {}, {}
    for name, param in model.named_parameters():
        if param.requires_grad and param.ndim >= 2 and (".bias" not in name):
            muon_params[name] = param
        else:
            adamw_params[name] = param

    for name, param in model.named_parameters():
        if param.requires_grad:
            if (param.ndim >= 2) or ((".bias" not in name) or (name.replace(".bias", ".weight") in muon_params)):
                # if name has bias, concatenate it with the .weight corresponding to the same layer
                if ".bias" in name:
                    weight_param = muon_params[name.replace(".bias", ".weight")]
                    bias_param = param
                    muon_params[name.replace(".bias", ".weight")] = [weight_param, bias_param]
                else:
                    muon_params[name] = param
            else:
                adamw_params[name] = param
    optimizer = Muon(
        muon_params=list(muon_params.values()),
        lr=learning_rate*4,
        adamw_params=list(adamw_params.values()),
        adamw_lr=learning_rate,
        adamw_wd=0,
        collective_whitening=True
    )
    error = model(torch.randn(1, 4, 32, 32).to(device), torch.zeros(1, device=device), torch.randint(0, 1000, (1,)).to(device))
    # print(error.shape)/
    error = error.sum()
    error.backward()
    optimizer.step()


if __name__ == "__main__":
    main()