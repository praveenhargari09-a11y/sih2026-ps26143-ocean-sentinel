import os
import shutil
import json
import requests
from pathlib import Path
from tqdm import tqdm
import py7zr
import rasterio
import numpy as np

PART3_URL = "https://zenodo.org/api/records/13761290/files/02_Test_images_and_ground_truth.7z/content"
DATA_ROOT = Path("data/raw/sar_oil_spill")
PILOT_ROOT = DATA_ROOT / "pilot"
ARCHIVE_PATH = DATA_ROOT / "downloads" / "02_Test_images_and_ground_truth.7z"

def download_file(url, filepath, max_retries=10):
    filepath.parent.mkdir(parents=True, exist_ok=True)
    
    for attempt in range(max_retries):
        try:
            initial_pos = 0
            headers = {}
            mode = "wb"
            
            response_head = requests.head(url, allow_redirects=True)
            total_size = int(response_head.headers.get("Content-Length", 0))
            
            if filepath.exists():
                initial_pos = filepath.stat().st_size
                if initial_pos >= total_size and total_size > 0:
                    print(f"File {filepath.name} already fully downloaded.")
                    return
                print(f"Resuming download from byte {initial_pos}...")
                headers["Range"] = f"bytes={initial_pos}-"
                mode = "ab"

            print(f"Downloading {filepath.name} (Attempt {attempt+1}/{max_retries})...")
            response = requests.get(url, headers=headers, stream=True, allow_redirects=True)
            response.raise_for_status()
            
            # If server doesn't support Range, it will return 200 instead of 206
            if response.status_code == 200 and initial_pos > 0:
                print("Server doesn't support resume. Restarting from scratch.")
                mode = "wb"
                initial_pos = 0
            
            with open(filepath, mode) as f, tqdm(
                desc=filepath.name, initial=initial_pos, total=total_size, unit="iB", unit_scale=True
            ) as pbar:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        size = f.write(chunk)
                        pbar.update(size)
            print("Download completed successfully!")
            return
        except Exception as e:
            print(f"\nError during download: {e}. Retrying in 5 seconds...")
            import time
            time.sleep(5)
            
    print("Failed to download file after maximum retries.")

def setup_pilot():
    # download_file(PART3_URL, ARCHIVE_PATH)
    
    extract_dir = DATA_ROOT / "downloads" / "extracted_part3"
    if not extract_dir.exists():
        print("Extracting archive...")
        extract_dir.mkdir(parents=True, exist_ok=True)
        # with py7zr.SevenZipFile(ARCHIVE_PATH, mode="r") as z:
        #     z.extractall(path=extract_dir)
            
    print("Organizing pilot dataset...")
    img_root = None
    mask_root = None
    for p in extract_dir.rglob("*"):
        if p.is_dir() and p.name.lower() == "images":
            img_root = p
        if p.is_dir() and p.name.lower() in ["mask", "masks"]:
            mask_root = p
            
    if not img_root or not mask_root:
        print("Could not find Images or Mask directories in the extracted archive.")
        return
        
    train_val_dir = PILOT_ROOT / "train_val"
    test_dir = PILOT_ROOT / "test"
    
    class_map = {
        "oil": ["Oil", "oil"],
        "lookalike": ["Lookalike", "lookalike", "Look-alike"],
        "no_oil": ["No oil", "no oil", "no_oil"]
    }
    
    for target_cat, source_names in class_map.items():
        (train_val_dir / target_cat / "images").mkdir(parents=True, exist_ok=True)
        (train_val_dir / target_cat / "masks").mkdir(parents=True, exist_ok=True)
        (test_dir / target_cat / "images").mkdir(parents=True, exist_ok=True)
        (test_dir / target_cat / "masks").mkdir(parents=True, exist_ok=True)
        
        src_img_dir = None
        src_mask_dir = None
        for name in source_names:
            if (img_root / name).exists(): src_img_dir = img_root / name
            if (mask_root / name).exists(): src_mask_dir = mask_root / name
            
        if not src_img_dir or not src_mask_dir:
            print(f"Warning: Could not find source dirs for {target_cat}")
            continue
            
        images = sorted(list(src_img_dir.glob("*.tif")))
        masks = sorted(list(src_mask_dir.glob("*.tif")))
        
        mask_map = {m.stem.replace('_mask', '').replace('_segmentation', ''): m for m in masks}
        pairs = []
        for img in images:
            stem = img.stem
            if stem in mask_map:
                pairs.append((img, mask_map[stem]))
                
        print(f"Found {len(pairs)} pairs for {target_cat}")
        
        for i, (img, mask) in enumerate(pairs):
            if i < 100:
                dest = train_val_dir
            else:
                dest = test_dir
                
            shutil.move(img, dest / target_cat / "images" / img.name)
            shutil.move(mask, dest / target_cat / "masks" / mask.name)

    print("Computing stats...")
    compute_stats(PILOT_ROOT)

def compute_stats(data_root):
    images = list((data_root / "train_val").rglob("images/*.tif"))
    if not images:
        print("No images found for stats.")
        return
        
    sample_size = min(50, len(images))
    samples = np.random.choice(images, sample_size, replace=False)
    
    all_band1 = []
    all_band2 = []
    band_count = 0
    
    for img_path in tqdm(samples, desc="Reading samples"):
        with rasterio.open(img_path) as src:
            data = src.read()
            band_count = data.shape[0]
            if band_count >= 1: all_band1.append(data[0].flatten())
            if band_count >= 2: all_band2.append(data[1].flatten())
            
    b1_data = np.concatenate(all_band1)
    b2_data = np.concatenate(all_band2) if all_band2 else None
    
    stats = {
        "band_count": int(band_count),
        "band1_min": float(np.min(b1_data)),
        "band1_max": float(np.max(b1_data)),
        "band1_p1": float(np.percentile(b1_data, 1)),
        "band1_p99": float(np.percentile(b1_data, 99)),
    }
    
    if b2_data is not None:
        stats.update({
            "band2_min": float(np.min(b2_data)),
            "band2_max": float(np.max(b2_data)),
            "band2_p1": float(np.percentile(b2_data, 1)),
            "band2_p99": float(np.percentile(b2_data, 99)),
        })
        db_min = min(stats["band1_p1"], stats["band2_p1"])
        db_max = max(stats["band1_p99"], stats["band2_p99"])
    else:
        db_min = stats["band1_p1"]
        db_max = stats["band1_p99"]
        
    stats["db_clip_range"] = [db_min, db_max]
    
    class_counts = {0: 0, 1: 0, 2: 0}
    
    for cat in ["oil", "lookalike", "no_oil"]:
        mask_dir = data_root / "train_val" / cat / "masks"
        if not mask_dir.exists(): continue
        
        for mask_path in tqdm(list(mask_dir.glob("*.tif")), desc=f"{cat} masks"):
            with rasterio.open(mask_path) as src:
                mask = src.read(1)
                
            if cat == "oil":
                fg = np.sum(mask > 0)
                class_counts[1] += fg
                class_counts[0] += (mask.size - fg)
            elif cat == "lookalike":
                fg = np.sum(mask > 0)
                class_counts[2] += fg
                class_counts[0] += (mask.size - fg)
            elif cat == "no_oil":
                class_counts[0] += mask.size
                
    stats["class_pixel_counts"] = {k: int(v) for k, v in class_counts.items()}
    
    total_pixels = sum(class_counts.values())
    if total_pixels > 0:
        weights = []
        for i in range(3):
            count = class_counts[i]
            if count > 0:
                weights.append(total_pixels / (3.0 * count))
            else:
                weights.append(0.0)
        stats["class_weights"] = weights
    else:
        stats["class_weights"] = [1.0, 1.0, 1.0]
        
    with open(data_root / "dataset_stats.json", "w") as f:
        json.dump(stats, f, indent=2)
        
    print(f"Stats saved to {data_root / 'dataset_stats.json'}")
    print(json.dumps(stats, indent=2))

if __name__ == '__main__':
    compute_stats(PILOT_ROOT)
