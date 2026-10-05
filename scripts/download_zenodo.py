import argparse
import json
import logging
import random
import shutil
import subprocess
from pathlib import Path

import numpy as np
import rasterio
import requests
from tqdm import tqdm

try:
    import py7zr
    HAS_PY7ZR = True
except ImportError:
    HAS_PY7ZR = False

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

URLS = {
    "01_Train_Val_Oil_Spill_images.7z": "https://zenodo.org/api/records/8346860/files/01_Train_Val_Oil_Spill_images.7z/content",
    "01_Train_Val_Oil_Spill_mask.7z": "https://zenodo.org/api/records/8346860/files/01_Train_Val_Oil_Spill_mask.7z/content",
    "01_Train_Val_No_Oil_Images.7z": "https://zenodo.org/api/records/8253899/files/01_Train_Val_No_Oil_Images.7z/content",
    "01_Train_Val_No_Oil_mask.7z": "https://zenodo.org/api/records/8253899/files/01_Train_Val_No_Oil_mask.7z/content",
    "01_Train_Val_Lookalike_images.7z": "https://zenodo.org/api/records/8253899/files/01_Train_Val_Lookalike_images.7z/content",
    "01_Train_Val_Lookalike_mask.7z": "https://zenodo.org/api/records/8253899/files/01_Train_Val_Lookalike_mask.7z/content",
    "02_Test_images_and_ground_truth.7z": "https://zenodo.org/api/records/13761290/files/02_Test_images_and_ground_truth.7z/content"
}


def check_sizes():
    logger.info("Checking sizes of files to download...")
    total_size = 0
    for filename, url in URLS.items():
        response = requests.head(url, allow_redirects=True)
        size = int(response.headers.get("Content-Length", 0))
        total_size += size
        logger.info(f"{filename}: {size / (1024 * 1024):.2f} MB")
    logger.info(f"Total size to download: {total_size / (1024 * 1024 * 1024):.2f} GB")


def download_file(url, filepath):
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    
    headers = {}
    mode = "wb"
    initial_pos = 0
    
    response_head = requests.head(url, allow_redirects=True)
    total_size = int(response_head.headers.get("Content-Length", 0))
    
    if filepath.exists():
        initial_pos = filepath.stat().st_size
        if initial_pos >= total_size and total_size > 0:
            logger.info(f"File {filepath.name} already fully downloaded.")
            return
        logger.info(f"Resuming download of {filepath.name} from {initial_pos} bytes")
        headers["Range"] = f"bytes={initial_pos}-"
        mode = "ab"
    
    response = requests.get(url, headers=headers, stream=True, allow_redirects=True)
    response.raise_for_status()
    
    with open(filepath, mode) as f, tqdm(
        desc=filepath.name,
        initial=initial_pos,
        total=total_size,
        unit="iB",
        unit_scale=True,
        unit_divisor=1024,
    ) as pbar:
        for chunk in response.iter_content(chunk_size=8192):
            if chunk:
                size = f.write(chunk)
                pbar.update(size)


def extract_7z(archive_path, extract_dir):
    archive_path = Path(archive_path)
    extract_dir = Path(extract_dir)
    extract_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Extracting {archive_path.name} to {extract_dir}")
    if HAS_PY7ZR:
        try:
            with py7zr.SevenZipFile(archive_path, mode="r") as z:
                z.extractall(path=extract_dir)
            return
        except Exception as e:
            logger.warning(f"py7zr failed: {e}. Falling back to 7z CLI...")
            
    try:
        subprocess.run(["7z", "x", f"-o{extract_dir}", str(archive_path)], check=True, stdout=subprocess.DEVNULL)
    except FileNotFoundError:
        logger.error("7z CLI not found. Please install 7-zip or py7zr.")
        raise
    except subprocess.CalledProcessError as e:
        logger.error(f"7z CLI failed: {e}")
        raise


def move_files(src_dir, dst_dir, pattern="*.tif*"):
    src_dir = Path(src_dir)
    dst_dir = Path(dst_dir)
    dst_dir.mkdir(parents=True, exist_ok=True)
    
    count = 0
    for file in src_dir.rglob(pattern):
        if file.is_file():
            shutil.move(str(file), str(dst_dir / file.name))
            count += 1
    return count


def organize_dataset(extract_root, data_root):
    extract_root = Path(extract_root)
    data_root = Path(data_root)
    
    # Define targets
    targets = {
        "train_val/oil/images": data_root / "train_val" / "oil" / "images",
        "train_val/oil/masks": data_root / "train_val" / "oil" / "masks",
        "train_val/lookalike/images": data_root / "train_val" / "lookalike" / "images",
        "train_val/lookalike/masks": data_root / "train_val" / "lookalike" / "masks",
        "train_val/no_oil/images": data_root / "train_val" / "no_oil" / "images",
        "train_val/no_oil/masks": data_root / "train_val" / "no_oil" / "masks",
        "test/oil/images": data_root / "test" / "oil" / "images",
        "test/oil/masks": data_root / "test" / "oil" / "masks",
        "test/lookalike/images": data_root / "test" / "lookalike" / "images",
        "test/lookalike/masks": data_root / "test" / "lookalike" / "masks",
        "test/no_oil/images": data_root / "test" / "no_oil" / "images",
        "test/no_oil/masks": data_root / "test" / "no_oil" / "masks",
    }
    
    for t in targets.values():
        t.mkdir(parents=True, exist_ok=True)

    logger.info("Organizing Train/Val Oil files...")
    move_files(extract_root / "01_Train_Val_Oil_Spill_images", targets["train_val/oil/images"])
    move_files(extract_root / "01_Train_Val_Oil_Spill_mask", targets["train_val/oil/masks"])
    
    logger.info("Organizing Train/Val Lookalike files...")
    move_files(extract_root / "01_Train_Val_Lookalike_images", targets["train_val/lookalike/images"])
    move_files(extract_root / "01_Train_Val_Lookalike_mask", targets["train_val/lookalike/masks"])
    
    logger.info("Organizing Train/Val No-Oil files...")
    move_files(extract_root / "01_Train_Val_No_Oil_Images", targets["train_val/no_oil/images"])
    move_files(extract_root / "01_Train_Val_No_Oil_mask", targets["train_val/no_oil/masks"])

    logger.info("Organizing Test files...")
    # Part III has internal folders Images/Oil, Mask/Oil, etc.
    test_root = extract_root / "02_Test_images_and_ground_truth"
    
    move_files(test_root / "Images" / "Oil", targets["test/oil/images"])
    move_files(test_root / "Mask" / "Oil", targets["test/oil/masks"])
    
    move_files(test_root / "Images" / "Lookalike", targets["test/lookalike/images"])
    move_files(test_root / "Mask" / "Lookalike", targets["test/lookalike/masks"])
    
    move_files(test_root / "Images" / "No oil", targets["test/no_oil/images"])
    move_files(test_root / "Mask" / "No oil", targets["test/no_oil/masks"])
    
    logger.info("Dataset organized successfully.")


def compute_statistics(data_root):
    data_root = Path(data_root)
    logger.info("Computing dataset statistics...")
    
    # 1. Sample 50 random images from train_val
    train_val_dir = data_root / "train_val"
    all_images = list(train_val_dir.rglob("images/*.tif*"))
    
    if not all_images:
        logger.error("No images found in train_val directory.")
        return
        
    sample_size = min(50, len(all_images))
    sample_images = random.sample(all_images, sample_size)
    
    all_bands_data = []
    band_count = 0
    
    for img_path in sample_images:
        with rasterio.open(img_path) as src:
            data = src.read()
            band_count = src.count
            all_bands_data.append(data)
            
    # shape: (N, C, H, W) -> (C, N*H*W)
    all_bands_data = np.stack(all_bands_data, axis=0)
    C = all_bands_data.shape[1]
    flattened_bands = all_bands_data.reshape(C, -1)
    
    band_stats = []
    global_min, global_max = np.inf, -np.inf
    
    for c in range(C):
        band_data = flattened_bands[c]
        stats = {
            "min": float(np.min(band_data)),
            "max": float(np.max(band_data)),
            "mean": float(np.mean(band_data)),
            "std": float(np.std(band_data)),
            "percentiles": {
                "1": float(np.percentile(band_data, 1)),
                "5": float(np.percentile(band_data, 5)),
                "50": float(np.percentile(band_data, 50)),
                "95": float(np.percentile(band_data, 95)),
                "99": float(np.percentile(band_data, 99))
            }
        }
        band_stats.append(stats)
        
    db_clip_range = [
        float(np.percentile(flattened_bands, 1)),
        float(np.percentile(flattened_bands, 99))
    ]
    
    # 2. Count total pixels per class across ALL train_val masks
    logger.info("Computing class pixel counts...")
    class_counts = {0: 0, 1: 0, 2: 0} # 0: no_oil, 1: oil, 2: lookalike
    
    cat_dirs = ["oil", "lookalike", "no_oil"]
    for cat in cat_dirs:
        mask_dir = train_val_dir / cat / "masks"
        masks = list(mask_dir.rglob("*.tif*"))
        
        for mask_path in tqdm(masks, desc=f"Processing {cat} masks"):
            with rasterio.open(mask_path) as src:
                mask = src.read(1)
                
                # Remap classes:
                # Original: fg=1, bg=0
                # Target: oil fg->1, lookalike fg->2, no_oil->0 (bg is always 0)
                if cat == "oil":
                    class_counts[1] += int(np.sum(mask == 1))
                    class_counts[0] += int(np.sum(mask == 0))
                elif cat == "lookalike":
                    class_counts[2] += int(np.sum(mask == 1))
                    class_counts[0] += int(np.sum(mask == 0))
                elif cat == "no_oil":
                    class_counts[0] += mask.size # All pixels are no_oil or bg (which is no_oil)

    # Compute inverse-frequency weights
    total_pixels = sum(class_counts.values())
    class_weights = {}
    for k, v in class_counts.items():
        if v > 0:
            class_weights[k] = total_pixels / v
        else:
            class_weights[k] = 0.0
            
    # Normalize weights so they sum to 1.0 (or just divide by sum of weights)
    weight_sum = sum(class_weights.values())
    if weight_sum > 0:
        for k in class_weights:
            class_weights[k] = float(class_weights[k] / weight_sum)
            
    stats_dict = {
        "band_count": band_count,
        "band_stats": band_stats,
        "class_pixel_counts": {str(k): v for k, v in class_counts.items()},
        "class_weights": {str(k): v for k, v in class_weights.items()},
        "db_clip_range": db_clip_range
    }
    
    out_file = data_root / "dataset_stats.json"
    with open(out_file, "w") as f:
        json.dump(stats_dict, f, indent=4)
        
    logger.info(f"Statistics saved to {out_file}")


def main():
    parser = argparse.ArgumentParser(description="Download and prepare Sentinel-1 SAR Oil Spill dataset.")
    parser.add_argument("--data-root", type=str, default="data/raw/sar_oil_spill", help="Target dataset root directory.")
    parser.add_argument("--download-dir", type=str, default="data/raw/sar_oil_spill/downloads", help="Directory for downloaded archives.")
    parser.add_argument("--skip-download", action="store_true", help="Skip downloading files.")
    parser.add_argument("--skip-extract", action="store_true", help="Skip extracting files.")
    parser.add_argument("--stats-only", action="store_true", help="Skip download and extract, only compute stats.")
    args = parser.parse_args()

    data_root = Path(args.data_root)
    download_dir = Path(args.download_dir)
    extract_dir = download_dir / "extracted"

    if args.stats_only:
        compute_statistics(data_root)
        return

    if not args.skip_download:
        check_sizes()
        download_dir.mkdir(parents=True, exist_ok=True)
        for filename, url in URLS.items():
            download_file(url, download_dir / filename)

    if not args.skip_extract:
        for filename in URLS.keys():
            archive_path = download_dir / filename
            if archive_path.exists():
                extract_7z(archive_path, extract_dir / archive_path.stem)
            else:
                logger.warning(f"Archive {archive_path} not found for extraction.")
        
        organize_dataset(extract_dir, data_root)

    compute_statistics(data_root)


if __name__ == "__main__":
    main()
