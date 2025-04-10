import torch
import torch.nn as nn
import numpy as np
import math
from timm.models.vision_transformer import PatchEmbed, Attention, Mlp
import click
from vdit import SiT

def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)

class TimestepEmbedder(nn.Module):
    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        # https://github.com/openai/glide-text2im/blob/main/glide_text2im/nn.py
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq)
        return t_emb


class LabelEmbedder(nn.Module):
    def __init__(self, num_classes, hidden_size, dropout_prob):
        super().__init__()
        use_cfg_embedding = dropout_prob > 0
        self.embedding_table = nn.Embedding(num_classes + use_cfg_embedding, hidden_size)
        self.num_classes = num_classes
        self.dropout_prob = dropout_prob

    def token_drop(self, labels, force_drop_ids=None):
        if force_drop_ids is None:
            drop_ids = torch.rand(labels.shape[0], device=labels.device) < self.dropout_prob
        else:
            drop_ids = force_drop_ids == 1
        labels = torch.where(drop_ids, self.num_classes, labels)
        return labels

    def forward(self, labels, train, force_drop_ids=None):
        use_dropout = self.dropout_prob > 0
        if (train and use_dropout) or (force_drop_ids is not None):
            labels = self.token_drop(labels, force_drop_ids)
        embeddings = self.embedding_table(labels)
        return embeddings


class SiTBlock(nn.Module):
    """
    A single Transformer block that optionally accepts a skip tensor.
    If skip=True, we learn a linear projection over the concatenation of the current features x and skip.
    """
    def __init__(
        self,
        hidden_size,
        num_heads,
        mlp_ratio=4.0,
        qkv_bias=True,
        norm_layer=nn.LayerNorm,
        act_layer=nn.GELU,
        skip=False,
        use_checkpoint=False,
    ):
        super().__init__()
        self.norm1 = norm_layer(hidden_size, eps=1e-6)
        self.attn = Attention(
            dim=hidden_size,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
        )
        self.norm2 = norm_layer(hidden_size, eps=1e-6)

        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        self.mlp = Mlp(
            in_features=hidden_size,
            hidden_features=mlp_hidden_dim,
            act_layer=act_layer,
            drop=0.0,
        )

        # For injecting time or label embeddings (AdaLayerNorm style)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 6 * hidden_size, bias=True)
        )

        # Skip connection logic
        self.skip_linear = nn.Linear(2 * hidden_size, hidden_size) if skip else None
        self.use_checkpoint = use_checkpoint

    def forward(self, x, c, skip=None):
        if self.use_checkpoint:
            return torch.utils.checkpoint.checkpoint(self._forward, x, c, skip)
        else:
            return self._forward(x, c, skip)

    def _forward(self, x, c, skip=None):
        # If skip_linear exists, we do "concat + linear" just like the paper
        if self.skip_linear is not None and skip is not None:
            x = self.skip_linear(torch.cat([x, skip], dim=-1))

        # AdaLayerNorm modulations from c
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(6, dim=1)

        # --- Attention path ---
        x_attn_normed = modulate(self.norm1(x), shift_msa, scale_msa)
        x_attn = self.attn(x_attn_normed)
        x = x + gate_msa.unsqueeze(1) * x_attn

        # --- MLP path ---
        x_mlp_normed = modulate(self.norm2(x), shift_mlp, scale_mlp)
        x_mlp = self.mlp(x_mlp_normed)
        x = x + gate_mlp.unsqueeze(1) * x_mlp

        return x
    

class FinalLayer(nn.Module):
    def __init__(self, hidden_size, patch_size, out_channels):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, patch_size * patch_size * out_channels, bias=True)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 2 * hidden_size, bias=True)
        )

    def forward(self, x, c):
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)
        x = modulate(self.norm_final(x), shift, scale)
        x = self.linear(x)
        return x

class GMMFinalLayer(nn.Module):
    def __init__(self, hidden_size, patch_size, in_channels, num_gaussians=8, constant_logstd=None):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        
        # Output means for K Gaussian components
        self.means_linear = nn.Linear(
            hidden_size, 
            patch_size * patch_size * num_gaussians * in_channels, 
            bias=True
        )
        
        # Output logweights for K Gaussian components
        self.logweights_linear = nn.Linear(
            hidden_size,
            patch_size * patch_size * num_gaussians,
            bias=True
        )
        
        # Conditioning modulation
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 2 * hidden_size, bias=True)
        )
        
        # Optional learned logstd
        self.constant_logstd = constant_logstd
        if constant_logstd is None:
            self.logstd_mlp = nn.Sequential(
                nn.SiLU(),
                nn.Linear(hidden_size, 1024),
                nn.SiLU(),
                nn.Linear(1024, 1)
            )
    
    def forward(self, x, c):
        # Apply AdaLN modulation
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)
        x = modulate(self.norm_final(x), shift, scale)
        
        # Get means and logweights
        means = self.means_linear(x)
        logweights = self.logweights_linear(x)
        
        # Get logstds
        if self.constant_logstd is None:
            logstds = self.logstd_mlp(c)
        else:
            logstds = torch.full((c.shape[0], 1), self.constant_logstd, device=c.device)
            
        return means, logweights, logstds

@torch.jit.script
def probabilistic_guidance_jit(cond_mean, total_var, uncond_mean, guidance_scale: float, orthogonal: float = 1.0):
    bias = cond_mean - uncond_mean
    if orthogonal > 0.0:
        bias = bias - ((bias * cond_mean).mean(
            dim=(-3, -2, -1), keepdim=True
        ) / (cond_mean * cond_mean).mean(dim=(-3, -2, -1), keepdim=True).clamp(min=1e-6) * cond_mean).mul(orthogonal)
    bias_power = (bias * bias).mean(dim=(-3, -2, -1), keepdim=True)
    avg_var = total_var.mean(dim=(-3, -2, -1), keepdim=True)
    bias = bias * ((avg_var / bias_power.clamp(min=1e-6)).sqrt() * guidance_scale)
    gaussian_output = dict(
        mean=cond_mean + bias,
        var=total_var * (1 - (guidance_scale * guidance_scale)))
    return gaussian_output, bias, avg_var


class GMSiT(SiT):
    """
    GMFlow implementation of SiT model
    """
    def __init__(
        self,
        input_size=32,
        patch_size=2,
        in_channels=4,
        hidden_size=1152,
        depth=28,
        num_heads=16,
        mlp_ratio=4.0,
        class_dropout_prob=0.1,
        num_classes=1000,
        num_gaussians=8,
        constant_logstd=None,
        use_spectral_sampling=False
    ):
        # Initialize with parent init but override learn_sigma
        super().__init__(
            input_size, patch_size, in_channels, hidden_size, 
            depth, num_heads, mlp_ratio, class_dropout_prob, 
            num_classes, learn_sigma=False
        )
        
        self.num_gaussians = num_gaussians
        self.use_spectral_sampling = use_spectral_sampling
        
        # Replace the original final layer with GMM final layer
        self.final_layer = GMMFinalLayer(
            hidden_size, 
            patch_size, 
            in_channels, 
            num_gaussians,
            constant_logstd
        )

        if use_spectral_sampling:
            self.spectrum_mlp = nn.Sequential(
                nn.Linear(2, 64),
                nn.SiLU(),
                nn.Linear(64, 64),
                nn.SiLU(),
                nn.Linear(64, input_size * input_size * in_channels)
            )
        
        # Re-initialize the weights for the new layer
        self._init_gmm_weights()
    
    def _init_gmm_weights(self):
        # Zero-out GMM-specific final layer
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.means_linear.weight, 0)
        nn.init.constant_(self.final_layer.means_linear.bias, 0)
        nn.init.constant_(self.final_layer.logweights_linear.weight, 0)
        nn.init.constant_(self.final_layer.logweights_linear.bias, 0)
        
        if self.final_layer.constant_logstd is None:
            nn.init.constant_(self.final_layer.logstd_mlp[-1].weight, 0)
            nn.init.constant_(self.final_layer.logstd_mlp[-1].bias, 0)
            
        if self.use_spectral_sampling:
            nn.init.constant_(self.spectrum_mlp[-1].weight, 0)
            nn.init.constant_(self.spectrum_mlp[-1].bias, 0)

    def unpatchify_gm(self, means, logweights, logstds):
        """Reshape tensor from (B, L, D) to (B, K, C, H, W)"""
        p = self.x_embedder.patch_size[0]
        h = w = int(means.shape[1] ** 0.5)
        assert h * w == means.shape[1]
        
        # Reshape means
        means = means.view(means.shape[0], h, w, self.num_gaussians, self.in_channels, p, p)
        means = means.permute(0, 3, 4, 1, 5, 2, 6).reshape(
            means.shape[0], self.num_gaussians, self.in_channels, h*p, w*p)
        
        # Reshape logweights
        logweights = logweights.view(logweights.shape[0], h, w, self.num_gaussians, p, p)
        logweights = logweights.permute(0, 3, 1, 4, 2, 5).reshape(
            logweights.shape[0], self.num_gaussians, 1, h*p, w*p)
        
        # Apply softmax to logweights across mixture dimension
        logweights = torch.log_softmax(logweights, dim=1)
        
        # Reshape logstds
        logstds = logstds.view(logstds.shape[0], 1, 1, 1, 1)
        
        return means, logweights, logstds
    
    def forward(self, x, t, y=None):
        # Process through the encoder-decoder backbone
        x_embed = self.x_embedder(x) + self.pos_embed
        t_emb = self.t_embedder(t)
        y_emb = self.y_embedder(y, self.training) if y is not None else 0
        c = t_emb + y_emb
        
        # Process through encoder-decoder blocks
        skips = []
        for blk in self.in_blocks:
            x_embed = blk(x_embed, c)
            skips.append(x_embed)
            
        x_embed = self.mid_block(x_embed, c)
        
        for blk in self.out_blocks:
            skip_x = skips.pop()
            x_embed = blk(x_embed, c, skip=skip_x)
        
        # GMM final layer
        means, logweights, logstds = self.final_layer(x_embed, c)
        means, logweights, logstds = self.unpatchify_gm(means, logweights, logstds)
        
        # Calculate derived statistics
        weights = torch.exp(logweights)
        mean = (weights * means).sum(dim=1)  # Mixture mean
        # Calculate variance: first term is the variance of each Gaussian
        # second term is the variance due to means being different from the mixture mean
        # weights shape: [B, K, 1, H, W], means shape: [B, K, C, H, W], mean shape: [B, C, H, W]
        # We need to sum over the mixture dimension (dim=1)
        var =(torch.exp(2 * logstds) + (weights * (means - mean.unsqueeze(1))**2)).sum(dim=1, keepdim=False)
        # print(f"mean shape: {mean.shape}, var shape: {var.shape}")
        
        # Return all GM parameters and statistics
        return {
            'means': means,
            'logweights': logweights,
            'logstds': logstds,
            'mean': mean,#.squeeze(1),
            'var': var#.squeeze(1)
        }
        
    def forward_with_cfg(self, x, t, y, cfg_scale):
        """Probabilistic guidance for GMFlow"""
        # Get conditional and unconditional outputs
        batch_size = y.shape[0]
        y_uncond = torch.full_like(y, 1000)
        # print(x.shape, t.shape, y.shape, y_uncond.shape)

        # print(x.shape, t.shape, y.shape)
        
        # Run conditional and unconditional forward passes
        cond_output = self.forward(x, t, y)
        uncond_output = self.forward(x, t, y_uncond)
        
        
        cond_mean = cond_output['mean']
        uncond_mean = uncond_output['mean']#.squeeze(1)
        cond_var = cond_output['var']#.squeeze(1)


        # invert cfg scale (1...4) to guidance scale (0...1)
        guidance_scale = min((cfg_scale - 1) / 4, 1)

        # print("cmean= ", cond_mean.shape, "| cvar= ", cond_var.shape, "| umean= ", uncond_mean.shape)
        
        guided_output, _, _ = probabilistic_guidance_jit(cond_mean, cond_var, uncond_mean, guidance_scale)

        guided_mean = guided_output['mean']#.squeeze(1)
        guided_var = guided_output['var']#.squeeze(1)

        # print("gmean= ", guided_mean.shape, "| gvar= ", guided_var.shape)
        
        # Return updated distribution
        # guided_output = {
        #     'mean': guided_mean,
        #     'var': guided_var
        # }
        
        return dict(
            mean=guided_mean,
            var=guided_var
        )




def get_2d_sincos_pos_embed(embed_dim, grid_size, cls_token=False, extra_tokens=0):
    grid_h = np.arange(grid_size, dtype=np.float32)
    grid_w = np.arange(grid_size, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)  # here w goes first
    grid = np.stack(grid, axis=0)

    grid = grid.reshape([2, 1, grid_size, grid_size])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token and extra_tokens > 0:
        pos_embed = np.concatenate([np.zeros([extra_tokens, embed_dim]), pos_embed], axis=0)
    return pos_embed


def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 2 == 0

    emb = np.concatenate([get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0]),  get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1]) ], axis=1) # (H*W, D)
    return emb


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    assert embed_dim % 2 == 0
    omega = 1. / (10000 ** ((np.arange(embed_dim // 2, dtype=np.float64)) / (embed_dim / 2.)))
    out = np.einsum('m,d->md', pos.reshape(-1), omega)
    emb = np.concatenate([np.sin(out), np.cos(out) ], axis=1)  
    return emb

#################################################################################
#                                   SiT Configs                                  #
#################################################################################
# Define the model variants
GMSiT_models = {
    'GMSiT-XL/2': lambda **kwargs: GMSiT(depth=28, hidden_size=1152, patch_size=2, num_heads=16, **kwargs),
    'GMSiT-XL/4': lambda **kwargs: GMSiT(depth=28, hidden_size=1152, patch_size=4, num_heads=16, **kwargs),
    'GMSiT-XL/8': lambda **kwargs: GMSiT(depth=28, hidden_size=1152, patch_size=8, num_heads=16, **kwargs),
    'GMSiT-L/2':  lambda **kwargs: GMSiT(depth=24, hidden_size=1024, patch_size=2, num_heads=16, **kwargs),
    'GMSiT-L/4':  lambda **kwargs: GMSiT(depth=24, hidden_size=1024, patch_size=4, num_heads=16, **kwargs),
    'GMSiT-L/8':  lambda **kwargs: GMSiT(depth=24, hidden_size=1024, patch_size=8, num_heads=16, **kwargs),
    'GMSiT-B/2':  lambda **kwargs: GMSiT(depth=12, hidden_size=768, patch_size=2, num_heads=12, **kwargs),
    'GMSiT-B/4':  lambda **kwargs: GMSiT(depth=12, hidden_size=768, patch_size=4, num_heads=12, **kwargs),
    'GMSiT-B/8':  lambda **kwargs: GMSiT(depth=12, hidden_size=768, patch_size=8, num_heads=12, **kwargs),
    'GMSiT-S/2':  lambda **kwargs: GMSiT(depth=12, hidden_size=384, patch_size=2, num_heads=6, **kwargs),
    'GMSiT-S/4':  lambda **kwargs: GMSiT(depth=12, hidden_size=384, patch_size=4, num_heads=6, **kwargs),
    'GMSiT-S/8':  lambda **kwargs: GMSiT(depth=12, hidden_size=384, patch_size=8, num_heads=6, **kwargs),
    'GMSiT-T/8':  lambda **kwargs: GMSiT(depth=6, hidden_size=192, patch_size=8, num_heads=6, **kwargs)
}

if __name__ == "__main__":
    model = GMSiT_models['GMSiT-XL/2'](use_spectral_sampling=True)
    print(model)
    x = torch.randn(8, 4, 32, 32)
    t = torch.randint(0, 1000, (8,))
    y = torch.randint(0, 1000, (8,))
    output = model.forward_with_cfg(x, t, y, cfg_scale=1.5)
    mean, var = output['mean'], output['var']
    print(mean.shape, var.shape)