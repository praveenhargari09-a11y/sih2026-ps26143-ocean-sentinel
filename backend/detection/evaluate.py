import argparse
import json
import numpy as np
import rasterio
import torch
import torch.nn.functional as F
from pathlib import Path
from tqdm import tqdm
from loguru import logger

# Import model and dataset utilities (assuming standard project structure based on instructions)
from backend.detection.model import load_model, OilSpillUNet
from backend.detection.dataset import build_test_set, load_dataset_stats


def get_gaussian_window(size: int, sigma: float = 0.5) -> np.ndarray:
    """Create a 2D Gaussian window for blending overlap regions."""
    x = np.linspace(-1, 1, size)
    y = np.linspace(-1, 1, size)
    xx, yy = np.meshgrid(x, y)
    d = np.sqrt(xx**2 + yy**2)
    window = np.exp(-(d**2) / (2.0 * sigma**2))
    return window


def tiled_inference(model, image, tile_size=512, overlap=128, device='cpu'):
    """
    Run model on overlapping tiles and stitch predictions.
    
    Args:
        model: the segmentation model
        image: numpy array (H, W, C) normalized
        tile_size: tile dimension (must match training crop)
        overlap: overlap between adjacent tiles in pixels
        device: torch device
    
    Returns:
        pred_map: (H, W) int64 class predictions
        prob_map: (H, W, num_classes) float32 class probabilities
    """
    h, w, c = image.shape
    stride = tile_size - overlap
    
    # Calculate padding for bottom and right edges
    pad_h = (tile_size - (h % stride)) % stride
    if (h - tile_size) % stride != 0:
        pad_h = stride - ((h - tile_size) % stride)
        
    pad_w = (tile_size - (w % stride)) % stride
    if (w - tile_size) % stride != 0:
        pad_w = stride - ((w - tile_size) % stride)

    # Apply reflection padding
    padded_image = np.pad(image, ((0, pad_h), (0, pad_w), (0, 0)), mode='reflect')
    pad_h_actual, pad_w_actual, _ = padded_image.shape
    
    num_classes = 3 # Background, Oil, Lookalike
    probs = np.zeros((pad_h_actual, pad_w_actual, num_classes), dtype=np.float32)
    weights = np.zeros((pad_h_actual, pad_w_actual, num_classes), dtype=np.float32)
    
    # Pre-compute Gaussian kernel
    kernel = get_gaussian_window(tile_size)
    kernel = np.expand_dims(kernel, axis=-1)
    kernel_torch = torch.from_numpy(kernel).to(device)
    
    model.eval()
    
    with torch.no_grad():
        with torch.amp.autocast(device_type='cuda' if 'cuda' in str(device) else 'cpu'):
            for y in range(0, pad_h_actual - tile_size + 1, stride):
                for x in range(0, pad_w_actual - tile_size + 1, stride):
                    tile = padded_image[y:y+tile_size, x:x+tile_size, :]
                    
                    # Convert to tensor: shape (1, C, H, W)
                    tile_tensor = torch.from_numpy(tile).permute(2, 0, 1).unsqueeze(0).float().to(device)
                    
                    # Forward pass
                    output = model(tile_tensor)
                    
                    # Get probabilities using softmax
                    prob = torch.softmax(output, dim=1).squeeze(0).permute(1, 2, 0) # Shape: (H, W, C)
                    
                    # Accumulate weighted probabilities
                    weighted_prob = (prob * kernel_torch).cpu().numpy()
                    probs[y:y+tile_size, x:x+tile_size, :] += weighted_prob
                    weights[y:y+tile_size, x:x+tile_size, :] += kernel
                    
    # Crop back to original image size
    probs = probs[:h, :w, :]
    weights = weights[:h, :w, :]
    
    # Finalize probabilities and predictions
    final_probs = probs / (weights + 1e-8)
    final_preds = np.argmax(final_probs, axis=-1).astype(np.int64)
    
    return final_preds, final_probs


def fast_hist(a, b, n):
    """Compute confusion matrix for a single image."""
    k = (a >= 0) & (a < n)
    return np.bincount(n * a[k].astype(int) + b[k], minlength=n ** 2).reshape(n, n)


def normalize_image(image_path, stats):
    """Read and normalize image using rasterio and db_clip stats."""
    with rasterio.open(image_path) as src:
        image = src.read() # Shape: (C, H, W)
        
    image = np.transpose(image, (1, 2, 0)) # Shape: (H, W, C)
    
    # Apply db_clip
    db_min = stats.get('db_min', -30.0)
    db_max = stats.get('db_max', 0.0)
    
    image = np.clip(image, db_min, db_max)
    image = (image - db_min) / (db_max - db_min)
    return image


def load_and_remap_mask(mask_path, item_metadata):
    """Load mask and apply 3-class remap (0: background, 1: oil, 2: lookalike)."""
    with rasterio.open(mask_path) as src:
        mask = src.read(1) # Shape: (H, W)
        
    mask = mask.astype(np.int64)
    
    # Determine class from metadata or path
    item_str = str(item_metadata).lower()
    path_str = str(mask_path).lower()
    
    if 'lookalike' in item_str or 'lookalike' in path_str:
        mask[mask == 1] = 2
    elif 'no_oil' in item_str or 'no_oil' in path_str:
        mask[mask == 1] = 0 # Safety check, no_oil should have no foreground
        
    return mask


def main():
    parser = argparse.ArgumentParser(description="Evaluate SAR Oil Spill Detection Model")
    parser.add_argument('--data-root', type=str, required=True, help="Path to raw dataset")
    parser.add_argument('--checkpoint', type=str, required=True, help="Path to model checkpoint")
    parser.add_argument('--tile-size', type=int, default=512, help="Tile size for inference")
    parser.add_argument('--overlap', type=int, default=128, help="Overlap between tiles")
    parser.add_argument('--batch-size', type=int, default=1, help="Batch size (must be 1 for tiled inference)")
    args = parser.parse_args()

    if args.batch_size != 1:
        logger.warning(f"Forced batch-size to 1 for tiled inference (was {args.batch_size})")

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logger.info(f"Using device: {device}")

    # 1. Load Model
    logger.info(f"Loading checkpoint from: {args.checkpoint}")
    try:
        model = load_model(args.checkpoint)
    except Exception as e:
        logger.warning(f"load_model failed ({e}), attempting manual OilSpillUNet load.")
        model = OilSpillUNet(encoder_name='resnet50', in_channels=2, classes=3)
        state_dict = torch.load(args.checkpoint, map_location='cpu', weights_only=True)
        if 'model_state_dict' in state_dict:
            model.load_state_dict(state_dict['model_state_dict'])
        else:
            model.load_state_dict(state_dict)
    
    model = model.to(device)
    model.eval()

    # 2. Load Dataset & Stats
    logger.info("Loading test dataset and stats...")
    dataset_stats = load_dataset_stats(args.data_root)
    test_set = build_test_set(args.data_root) # Expected to yield 450 items
    
    logger.info(f"Total test samples: {len(test_set)}")

    num_classes = 3
    hist = np.zeros((num_classes, num_classes))

    # 3. Inference Loop
    logger.info("Starting evaluation...")
    for item in tqdm(test_set, desc="Evaluating"):
        # Handle dict or tuple format from build_test_set
        if isinstance(item, dict):
            img_path = item.get('image_path') or item.get('image')
            mask_path = item.get('mask_path') or item.get('mask')
        elif isinstance(item, (tuple, list)):
            img_path, mask_path = item[0], item[1]
        else:
            # Fallback if it's an object
            img_path = getattr(item, 'image_path', getattr(item, 'image', None))
            mask_path = getattr(item, 'mask_path', getattr(item, 'mask', None))
            
        if img_path is None or mask_path is None:
            logger.error(f"Could not parse paths from test set item: {item}")
            continue
            
        # Read & Normalize
        image = normalize_image(img_path, dataset_stats)
        
        # Read & Remap Mask
        true_mask = load_and_remap_mask(mask_path, item)
        
        # Tiled Inference
        pred_map, _ = tiled_inference(
            model, 
            image, 
            tile_size=args.tile_size, 
            overlap=args.overlap, 
            device=device
        )
        
        # Accumulate confusion matrix
        hist += fast_hist(true_mask.flatten(), pred_map.flatten(), num_classes)

    # 4. Compute Metrics
    epsilon = 1e-8
    # hist shape is (True, Predicted)
    # diag is true positives per class
    diag = np.diag(hist)
    sum_row = hist.sum(axis=1) # Actual / Support
    sum_col = hist.sum(axis=0) # Predicted

    iou = diag / (sum_row + sum_col - diag + epsilon)
    precision = diag / (sum_col + epsilon)
    recall = diag / (sum_row + epsilon)
    f1 = 2 * (precision * recall) / (precision + recall + epsilon)
    dice = 2 * diag / (sum_row + sum_col + epsilon)
    
    mean_iou = np.nanmean(iou)
    weighted_iou = np.nansum(iou * sum_row) / (np.sum(sum_row) + epsilon)
    
    # Normalized confusion matrix (by row = recall per class)
    cm_normalized = hist / (sum_row[:, None] + epsilon)
    
    # Specific targeted metrics
    # Lookalike (Class 2) predicted as Oil (Class 1)
    lookalike_as_oil = hist[2, 1]
    lookalike_total = sum_row[2]
    lookalike_suppression_rate = 1.0 - (lookalike_as_oil / (lookalike_total + epsilon))
    if lookalike_total == 0: lookalike_suppression_rate = 1.0
    
    # Oil (Class 1) correctly predicted as Oil (Class 1)
    oil_detected = hist[1, 1]
    oil_total = sum_row[1]
    oil_detection_rate = oil_detected / (oil_total + epsilon)
    
    # Background (Class 0) predicted as Oil (Class 1)
    bg_as_oil = hist[0, 1]
    bg_total = sum_row[0]
    false_alarm_rate = bg_as_oil / (bg_total + epsilon)

    class_names = ["Background", "Oil", "Lookalike"]

    # 5. Report & Save
    print("\n" + "="*50)
    print("EVALUATION RESULTS")
    print("="*50)
    print(f"{'Class':<12} | {'IoU':<6} | {'Dice':<6} | {'F1':<6} | {'Prec':<6} | {'Recall':<6} | {'Support':<10}")
    print("-" * 65)
    for i, name in enumerate(class_names):
        print(f"{name:<12} | {iou[i]:.4f} | {dice[i]:.4f} | {f1[i]:.4f} | {precision[i]:.4f} | {recall[i]:.4f} | {int(sum_row[i]):<10}")
    
    print("\nOVERALL METRICS:")
    print(f"Mean IoU:      {mean_iou:.4f}")
    print(f"Weighted IoU:  {weighted_iou:.4f}")
    
    print("\nTARGET METRICS:")
    print(f"Oil Detection Rate (Recall):     {oil_detection_rate:.4f} ({int(oil_detected)}/{int(oil_total)} px)")
    print(f"Lookalike Suppression Rate:      {lookalike_suppression_rate:.4f} (Suppressed {int(lookalike_total - lookalike_as_oil)}/{int(lookalike_total)} px)")
    print(f"False Alarm Rate (Bg -> Oil):    {false_alarm_rate:.6f} ({int(bg_as_oil)}/{int(bg_total)} px)")
    
    print("\nCONFUSION MATRIX (Normalized by row/true class):")
    print(f"{'True \\ Pred':<12} | {'Background':<10} | {'Oil':<10} | {'Lookalike':<10}")
    print("-" * 50)
    for i, name in enumerate(class_names):
        print(f"{name:<12} | {cm_normalized[i, 0]:.4f}     | {cm_normalized[i, 1]:.4f}     | {cm_normalized[i, 2]:.4f}")

    results_dict = {
        "per_class": {
            name: {
                "iou": float(iou[i]),
                "dice": float(dice[i]),
                "f1": float(f1[i]),
                "precision": float(precision[i]),
                "recall": float(recall[i]),
                "support": int(sum_row[i])
            } for i, name in enumerate(class_names)
        },
        "overall": {
            "mean_iou": float(mean_iou),
            "weighted_iou": float(weighted_iou),
            "lookalike_suppression_rate": float(lookalike_suppression_rate),
            "oil_detection_rate": float(oil_detection_rate),
            "false_alarm_rate": float(false_alarm_rate)
        },
        "confusion_matrix_raw": hist.tolist(),
        "confusion_matrix_normalized": cm_normalized.tolist()
    }

    # Save to json
    checkpoint_path = Path(args.checkpoint)
    out_json = checkpoint_path.parent / "test_results.json"
    out_json.parent.mkdir(parents=True, exist_ok=True)
    
    with open(out_json, "w") as f:
        json.dump(results_dict, f, indent=4)
        
    logger.info(f"Saved evaluation results to: {out_json}")

if __name__ == "__main__":
    main()
