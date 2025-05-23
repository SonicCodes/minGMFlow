import wandb
import requests
from PIL import Image
from io import BytesIO
from collections import defaultdict
import os
import moviepy as mpy
import numpy as np
import re
import concurrent.futures
import threading

# === CONFIGURATION ===

api = wandb.Api(timeout=1000)
entity = "lucidsim"
project = "lucid_vaes_v2"
run_id = "4s1c3b1w"
frame_duration_ms = 100
output_dir = f"videos/{run_id}"
os.makedirs(output_dir, exist_ok=True)
api.entity = entity
# === FETCH RUN ===
run = api.run(f"{entity}/{project}/{run_id}")

# === COLLECT MEDIA BY LOGGING KEY ===
media_by_key = defaultdict(list)

print("🔍 Scanning run history for media...")

for row in run.history(pandas=False, samples=1_000_000):
    step = row["_step"]
    print(step)
    for key, val in row.items():
        # print(key, type(val))
        if isinstance(val, dict): 
            print(val)
            if  ("filenames" in val) :
                fname = val["filenames"][12]
                if any(fname.lower().endswith(ext) for ext in [".png", ".jpg", ".jpeg"]):
                    if key not in media_by_key:
                        media_by_key[key] = []
                    media_by_key[key].append((step, fname))

# Function to download a single image
def download_image(url, run, cache_dir="./cache"):
    try:
        file = run.file(url[1]).download(replace=True, root=cache_dir)
        img = Image.open(f"{cache_dir}/{url[1]}").convert("RGB")
        return (url[0], np.array(img))
    except Exception as e:
        print(f"⚠️ Skipping image from {url}: {e}")
        return None

# === PROCESS EACH KEY INTO A MP4 ===
for key, step_urls in media_by_key.items():
    # Sort by step (smaller first)
    # step_urls.sort(key=lambda x: x[0])
    urls = step_urls
    urls = list(dict.fromkeys(urls))  # deduplicate while preserving order
    print(f"🎨 Creating MP4 for '{key}' with {len(urls)} frames...")

    # Parallelize image downloading
    images = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=16) as executor:
        future_to_url = {executor.submit(download_image, url, run): url for url in urls}
        for future in concurrent.futures.as_completed(future_to_url):
            img = future.result()
            if img is not None:
                images.append(img)

    images.sort(key=lambda x: x[0])

    if images:
        output_path = os.path.join(output_dir, f"{key}.mp4")
        fps = 1000 / frame_duration_ms
        
        # Create a clip from the images using moviepy
        clip = mpy.ImageSequenceClip([img[1] for img in images], fps=fps)
        clip.write_videofile(output_path, codec='libx264')
        
        print(f"✅ Saved: {output_path}")
    else:
        print(f"❌ No valid images found for key '{key}'.")

print("🏁 Done.")