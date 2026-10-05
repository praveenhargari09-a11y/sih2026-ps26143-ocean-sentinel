import os
import json
import glob
import random
import numpy as np
import rasterio
from pathlib import Path
from loguru import logger
import torch
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2
from sklearn.model_selection import train_test_split

def load_dataset_stats(data_root: str) -> dict:
    """
    Loads dataset_stats.json from the data_root directory.
    Returns a dictionary of stats, or an empty dictionary if not found.
    """
    stats_path = Path(data_root) / "dataset_stats.json"
    if stats_path.exists():
        try:
            with open(stats_path, 'r') as f:
                stats = json.load(f)
            logger.info(f"Loaded dataset stats from {stats_path}")
            return stats
        except Exception as e:
            logger.warning(f"Failed to load dataset stats: {e}")
            return {}
    else:
        logger.warning(f"No dataset_stats.json found at {stats_path}")
        return {}

def build_splits(data_root: str, val_fraction: float = 0.15, seed: int = 42) -> tuple[list, list]:
    """
    Walks through the train_val directory and creates stratified train/val splits.
    Returns:
        train_samples: List of (image_path, mask_path, category)
        val_samples: List of (image_path, mask_path, category)
    """
    random.seed(seed)
    np.random.seed(seed)
    
    train_val_dir = Path(data_root) / "train_val"
    if not train_val_dir.exists():
        raise FileNotFoundError(f"train_val directory not found at {train_val_dir}")
        
    categories = ["oil", "lookalike", "no_oil"]
    all_samples = []
    labels = []
    
    for cat in categories:
        img_dir = train_val_dir / cat / "images"
        mask_dir = train_val_dir / cat / "masks"
        
        if not img_dir.exists() or not mask_dir.exists():
            logger.warning(f"Missing images/masks directory for category {cat}")
            continue
            
        img_paths = sorted(glob.glob(str(img_dir / "*.tif")))
        mask_paths = sorted(glob.glob(str(mask_dir / "*.tif")))
        
        # Match by filename
        img_map = {Path(p).stem: p for p in img_paths}
        mask_map = {Path(p).stem.replace('_mask', '').replace('_segmentation', ''): p for p in mask_paths}
        
        matched = 0
        for name, img_p in img_map.items():
            if name in mask_map:
                all_samples.append((img_p, mask_map[name], cat))
                labels.append(cat)
                matched += 1
                
        logger.info(f"Found {matched} matching image-mask pairs for category '{cat}'")

    if not all_samples:
        return [], []
        
    train_samples, val_samples = train_test_split(
        all_samples, 
        test_size=val_fraction, 
        random_state=seed, 
        stratify=labels
    )
    
    logger.info(f"Split sizes: Train={len(train_samples)}, Val={len(val_samples)}")
    return train_samples, val_samples

def build_test_set(data_root: str) -> list:
    """
    Walks through the test directory and builds a list of test samples.
    Returns:
        test_samples: List of (image_path, mask_path, category)
    """
    test_dir = Path(data_root) / "test"
    if not test_dir.exists():
        logger.warning(f"Test directory not found at {test_dir}")
        return []
        
    categories = ["oil", "lookalike", "no_oil"]
    test_samples = []
    
    for cat in categories:
        img_dir = test_dir / cat / "images"
        mask_dir = test_dir / cat / "masks"
        
        if not img_dir.exists() or not mask_dir.exists():
            continue
            
        img_paths = sorted(glob.glob(str(img_dir / "*.tif")))
        mask_paths = sorted(glob.glob(str(mask_dir / "*.tif")))
        
        img_map = {Path(p).stem: p for p in img_paths}
        mask_map = {Path(p).stem.replace('_mask', '').replace('_segmentation', ''): p for p in mask_paths}
        
        for name, img_p in img_map.items():
            if name in mask_map:
                test_samples.append((img_p, mask_map[name], cat))
                
    logger.info(f"Found {len(test_samples)} test samples")
    return test_samples

class SARSpillDataset(Dataset):
    """
    PyTorch Dataset for Sentinel-1 SAR Oil Spill dataset.
    """
    def __init__(self, samples: list, transform=None, crop_size: int = 512, 
                 db_clip: tuple = (-30, 0), fg_crop_prob: float = 0.7, 
                 is_train: bool = True):
        """
        Args:
            samples: List of (image_path, mask_path, category) tuples.
            transform: Albumentations transforms to apply.
            crop_size: Size of the square crop (default 512).
            db_clip: Min/Max decibel values for clipping and normalization.
            fg_crop_prob: Probability of using foreground-aware cropping.
            is_train: If True, uses foreground/random crop. If False, uses center crop.
        """
        self.samples = samples
        self.transform = transform
        self.crop_size = crop_size
        self.db_clip = db_clip
        self.fg_crop_prob = fg_crop_prob
        self.is_train = is_train

    def __len__(self):
        return len(self.samples)

    def _get_crop_coords(self, h, w, mask, cat):
        """Helper to determine the bounding box for cropping"""
        ch, cw = self.crop_size, self.crop_size
        
        # If not training, just center crop
        if not self.is_train:
            y0 = max(0, (h - ch) // 2)
            x0 = max(0, (w - cw) // 2)
            return y0, x0
            
        # For training: Foreground-aware crop
        if cat != "no_oil" and random.random() < self.fg_crop_prob:
            y_indices, x_indices = np.where(mask > 0)
            if len(y_indices) > 0:
                # Pick a random foreground pixel
                idx = random.randint(0, len(y_indices) - 1)
                cy, cx = y_indices[idx], x_indices[idx]
                
                # Add random jitter
                cy += random.randint(-64, 64)
                cx += random.randint(-64, 64)
                
                # Clamp center to valid range for cropping
                cy = max(ch // 2, min(h - ch // 2, cy))
                cx = max(cw // 2, min(w - cw // 2, cx))
                
                y0 = cy - ch // 2
                x0 = cx - cw // 2
                return int(y0), int(x0)
                
        # Random crop fallback
        y0 = random.randint(0, max(0, h - ch))
        x0 = random.randint(0, max(0, w - cw))
        return y0, x0

    def __getitem__(self, idx):
        img_path, mask_path, cat = self.samples[idx]
        
        # Read Image
        with rasterio.open(img_path) as src:
            image = src.read() # (2, H, W)
            
        # Read Mask
        with rasterio.open(mask_path) as m_src:
            mask = m_src.read(1) # (H, W)
            
        # Re-arrange dimensions for Albumentations (H, W, 2)
        image = np.transpose(image, (1, 2, 0))
        h, w = image.shape[:2]
        
        # Format Mask: 3-class remap
        # Foreground is assumed to be > 0 in original mask
        remapped_mask = np.zeros_like(mask, dtype=np.int64)
        if cat == 'oil':
            remapped_mask[mask > 0] = 1
        elif cat == 'lookalike':
            remapped_mask[mask > 0] = 2
            
        # Crop logic
        y0, x0 = self._get_crop_coords(h, w, remapped_mask, cat)
        image_crop = image[y0 : y0 + self.crop_size, x0 : x0 + self.crop_size]
        mask_crop = remapped_mask[y0 : y0 + self.crop_size, x0 : x0 + self.crop_size]
        
        # Normalize Sigma0 dB to [0, 1]
        image_crop = np.clip(image_crop, self.db_clip[0], self.db_clip[1])
        db_range = self.db_clip[1] - self.db_clip[0]
        if db_range > 0:
            image_crop = (image_crop - self.db_clip[0]) / db_range
        image_crop = image_crop.astype(np.float32)
        
        # Apply transforms (e.g., flips, rotations, ToTensor)
        if self.transform:
            augmented = self.transform(image=image_crop, mask=mask_crop)
            image_crop = augmented['image']
            mask_crop = augmented['mask']
            
        # mask_crop should be int64 (CrossEntropyLoss expects Long type)
        if isinstance(mask_crop, torch.Tensor):
            mask_crop = mask_crop.to(torch.int64)
        else:
            mask_crop = torch.tensor(mask_crop, dtype=torch.int64)
            
        return {
            'image': image_crop,
            'mask': mask_crop,
            'category': cat,
            'filepath': img_path
        }

def get_dataloaders(data_root: str, batch_size: int = 4, val_fraction: float = 0.15, 
                    crop_size: int = 512, num_workers: int = 0, seed: int = 42, limit: int = None) -> tuple[DataLoader, DataLoader]:
    """
    Creates and returns PyTorch DataLoaders for train and val sets.
    """
    # Load dataset stats
    stats = load_dataset_stats(data_root)
    db_clip = tuple(stats.get("db_clip_range", [-30, 0]))
    
    # Build splits
    train_samples, val_samples = build_splits(data_root, val_fraction, seed)
    
    if limit is not None:
        import random
        random.seed(seed)
        random.shuffle(train_samples)
        random.shuffle(val_samples)
        train_samples = train_samples[:limit]
        val_samples = val_samples[:max(1, limit // 4)]
        
    logger.info(f"Using {len(train_samples)} train samples and {len(val_samples)} val samples")
    
    # Define augmentations (Applied AFTER cropping)
    train_transform = A.Compose([
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=15, p=0.3),
        ToTensorV2()
    ])
    
    val_transform = A.Compose([
        ToTensorV2()
    ])
    
    # Datasets
    train_dataset = SARSpillDataset(
        samples=train_samples,
        transform=train_transform,
        crop_size=crop_size,
        db_clip=db_clip,
        fg_crop_prob=0.7,
        is_train=True
    )
    
    val_dataset = SARSpillDataset(
        samples=val_samples,
        transform=val_transform,
        crop_size=crop_size,
        db_clip=db_clip,
        fg_crop_prob=0.0,
        is_train=False
    )
    
    # DataLoaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=num_workers,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=num_workers
    )
    
    return train_loader, val_loader

if __name__ == '__main__':
    # Simple CLI self-test
    # Replace data_root with a real path when running, or test if it exists
    test_data_root = "data/raw/sar_oil_spill"
    
    if Path(test_data_root).exists():
        logger.info("Running dataset self-test...")
        try:
            train_loader, val_loader = get_dataloaders(test_data_root, batch_size=2)
            
            logger.info(f"Train batches: {len(train_loader)}")
            logger.info(f"Val batches: {len(val_loader)}")
            
            if len(train_loader) > 0:
                batch = next(iter(train_loader))
                imgs = batch['image']
                masks = batch['mask']
                cats = batch['category']
                paths = batch['filepath']
                
                logger.info(f"Image batch shape: {imgs.shape}")
                logger.info(f"Mask batch shape: {masks.shape}")
                logger.info(f"Image min/max: {imgs.min():.2f} / {imgs.max():.2f}")
                logger.info(f"Categories in batch: {cats}")
                logger.info(f"Unique mask values: {torch.unique(masks)}")
            
        except Exception as e:
            logger.error(f"Test failed: {e}")
    else:
        logger.info(f"Mocking a dataset test since {test_data_root} doesn't exist.")
        # We can construct dummy samples and test SARSpillDataset logic
        dummy_img = np.random.uniform(-35, 5, (2, 2048, 2048)).astype(np.float32)
        dummy_mask = np.zeros((1, 2048, 2048), dtype=np.uint8)
        dummy_mask[0, 1000:1100, 1000:1100] = 1 # Fake foreground
        
        dummy_img_path = "dummy_img.tif"
        dummy_mask_path = "dummy_mask.tif"
        
        try:
            # Create dummy TIFFs
            with rasterio.open(dummy_img_path, 'w', driver='GTiff', width=2048, height=2048, count=2, dtype='float32') as dst:
                dst.write(dummy_img)
            with rasterio.open(dummy_mask_path, 'w', driver='GTiff', width=2048, height=2048, count=1, dtype='uint8') as dst:
                dst.write(dummy_mask)
                
            samples = [(dummy_img_path, dummy_mask_path, "oil")]
            dataset = SARSpillDataset(samples, transform=A.Compose([ToTensorV2()]), is_train=True)
            
            item = dataset[0]
            logger.info(f"Dummy Image shape: {item['image'].shape}")
            logger.info(f"Dummy Mask shape: {item['mask'].shape}")
            logger.info(f"Dummy Mask unique values: {torch.unique(item['mask'])}")
            
        finally:
            if os.path.exists(dummy_img_path): os.remove(dummy_img_path)
            if os.path.exists(dummy_mask_path): os.remove(dummy_mask_path)
