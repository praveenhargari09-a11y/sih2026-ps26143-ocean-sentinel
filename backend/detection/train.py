"""
Training script for SAR oil-spill segmentation.
"""
import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from loguru import logger

# Import from local modules
from backend.detection.model import OilSpillUNet, CombinedLoss
from backend.detection.dataset import get_dataloaders, load_dataset_stats

def compute_metrics(preds, targets, num_classes=3):
    """
    Compute per-class IoU, mIoU, Oil Dice, and confusion matrix metrics.
    preds: (B, H, W) integer class predictions
    targets: (B, H, W) integer ground truth
    """
    preds = preds.flatten()
    targets = targets.flatten()
    
    # Confusion matrix
    # Using bincount trick for 1D array of 0 to num_classes**2 - 1
    cm = torch.bincount(num_classes * targets + preds, minlength=num_classes**2).reshape(num_classes, num_classes)
    
    # Calculate IoU per class
    intersection = cm.diag()
    ground_truth_set = cm.sum(dim=1)
    predicted_set = cm.sum(dim=0)
    union = ground_truth_set + predicted_set - intersection
    
    iou = intersection.float() / (union.float() + 1e-6)
    
    # Oil Dice (Class 1)
    # Dice = 2 * TP / (2*TP + FP + FN)
    tp_oil = intersection[1].float()
    fp_oil = predicted_set[1].float() - tp_oil
    fn_oil = ground_truth_set[1].float() - tp_oil
    oil_dice = 2 * tp_oil / (2 * tp_oil + fp_oil + fn_oil + 1e-6)
    
    # Lookalike-as-oil rate: cm[2][1] / sum(cm[2])
    lookalike_total = ground_truth_set[2].float()
    lookalike_as_oil_rate = cm[2][1].float() / (lookalike_total + 1e-6) if lookalike_total > 0 else torch.tensor(0.0, device=cm.device)
    
    return {
        "iou": iou.cpu().numpy(),
        "miou": iou.mean().item(),
        "oil_dice": oil_dice.item(),
        "cm": cm.cpu().numpy(),
        "lookalike_as_oil_rate": lookalike_as_oil_rate.item()
    }

def main():
    parser = argparse.ArgumentParser(description="Train SAR oil-spill segmentation model")
    parser.add_argument("--data-root", type=str, default="data/raw/sar_oil_spill")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--crop-size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--val-fraction", type=float, default=0.15)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of training samples for fast micro-training")
    parser.add_argument("--encoder", type=str, default="resnet50", help="Model encoder (e.g., mobilenet_v2 for micro-training)")
    parser.add_argument("--resume", type=str, default=None)
    args = parser.parse_args()

    device = torch.device(args.device)
    logger.info(f"Using device: {device}")

    # Create checkpoint directory
    checkpoint_dir = Path("models/checkpoints")
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    csv_path = checkpoint_dir / "training_log.csv"
    csv_exists = csv_path.exists()
    
    # Load dataset stats
    stats = load_dataset_stats(args.data_root)
    class_weights = stats.get("class_weights", [1.0, 1.0, 1.0])
    band_count = stats.get("band_count", 2)
    logger.info(f"Dataset stats loaded. Band count: {band_count}, Class weights: {class_weights}")
    
    # Create Dataloaders
    train_loader, val_loader = get_dataloaders(
        data_root=args.data_root,
        batch_size=args.batch_size,
        crop_size=args.crop_size,
        val_fraction=args.val_fraction,
        num_workers=args.num_workers,
        limit=args.limit
    )
    logger.info(f"Dataloaders ready: {len(train_loader)} train batches, {len(val_loader)} val batches.")
    
    # Create Model
    model = OilSpillUNet(num_classes=3, encoder=args.encoder, in_channels=band_count).to(device)
    param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Model created (encoder: {args.encoder}). Trainable parameters: {param_count:,}")
    
    # Loss, Optimizer, Scheduler
    cw_tensor = torch.tensor(class_weights).float().to(device)
    criterion = CombinedLoss(ce_weight=0.5, dice_weight=0.5, class_weights=cw_tensor)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)
    
    scaler = torch.amp.GradScaler('cuda', enabled=(device.type == 'cuda'))
    
    start_epoch = 0
    best_oil_iou = -1.0
    
    # Resume from checkpoint if specified
    if args.resume and os.path.exists(args.resume):
        logger.info(f"Resuming from checkpoint: {args.resume}")
        checkpoint = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_oil_iou = checkpoint.get('val_oil_iou', -1.0)
        logger.info(f"Resumed at epoch {start_epoch}, best oil_iou so far: {best_oil_iou:.4f}")
    
    # Initialize CSV
    with open(csv_path, mode="a", newline="") as f:
        writer = csv.writer(f)
        if not csv_exists or start_epoch == 0:
            writer.writerow(["epoch", "train_loss", "val_loss", "val_miou", "val_oil_iou", "val_lookalike_iou", "val_bg_iou", "oil_dice", "lookalike_as_oil_rate", "lr"])
            
    logger.info("Starting training loop...")
    
    try:
        for epoch in range(start_epoch, args.epochs):
            # Training Phase
            model.train()
            train_loss_accum = 0.0
            
            for batch_idx, batch in enumerate(train_loader):
                images = batch['image'].to(device)
                masks = batch['mask'].to(device)
                
                optimizer.zero_grad()
                
                # Probe GPU Memory on very first batch
                probe_memory = (epoch == 0 and batch_idx == 0)
                if probe_memory:
                    start_time = time.perf_counter()
                
                with torch.amp.autocast('cuda', enabled=(device.type == 'cuda')):
                    outputs = model(images)
                    loss = criterion(outputs, masks)
                
                scaler.scale(loss).backward()
                
                if probe_memory:
                    torch.cuda.synchronize() if device.type == 'cuda' else None
                    forward_time_ms = (time.perf_counter() - start_time) * 1000
                    if device.type == 'cuda':
                        peak_mem_mb = torch.cuda.max_memory_allocated() / (1024 ** 2)
                        logger.info(f"GPU Memory: {peak_mem_mb:.2f} MB peak, Forward: {forward_time_ms:.2f} ms")
                    else:
                        logger.info(f"Running on CPU — no GPU memory data available, Forward: {forward_time_ms:.2f} ms")
                        
                    # Calculate foreground crop ratio for the batch
                    fg_crops = (masks > 0).view(masks.size(0), -1).sum(dim=1) > 0
                    fg_ratio = fg_crops.float().mean().item()
                    logger.info(f"Batch 0 foreground crop ratio: {fg_ratio:.2%} ({fg_crops.sum()}/{masks.size(0)} crops have fg)")
                
                scaler.step(optimizer)
                scaler.update()
                
                train_loss_accum += loss.item() * images.size(0)
            
            avg_train_loss = train_loss_accum / len(train_loader.dataset)
            
            # Validation Phase
            model.eval()
            val_loss_accum = 0.0
            
            # Metrics accumulation
            all_iou = []
            all_oil_dice = []
            all_cm = None
            
            with torch.no_grad():
                for batch in val_loader:
                    images = batch['image'].to(device)
                    masks = batch['mask'].to(device)
                    
                    with torch.amp.autocast('cuda', enabled=(device.type == 'cuda')):
                        outputs = model(images)
                        loss = criterion(outputs, masks)
                    
                    val_loss_accum += loss.item() * images.size(0)
                    
                    preds = torch.argmax(outputs, dim=1)
                    batch_metrics = compute_metrics(preds, masks, num_classes=3)
                    
                    all_iou.append(batch_metrics['iou'])
                    all_oil_dice.append(batch_metrics['oil_dice'])
                    if all_cm is None:
                        all_cm = batch_metrics['cm']
                    else:
                        all_cm += batch_metrics['cm']
            
            avg_val_loss = val_loss_accum / len(val_loader.dataset)
            
            # Aggregate metrics
            mean_iou_per_class = sum(all_iou) / len(all_iou)
            bg_iou, oil_iou, lookalike_iou = mean_iou_per_class
            val_miou = mean_iou_per_class.mean().item()
            mean_oil_dice = sum(all_oil_dice) / len(all_oil_dice)
            
            # Lookalike-as-oil rate over entire epoch
            lookalike_total = all_cm[2].sum()
            lookalike_as_oil_rate = (all_cm[2][1] / (lookalike_total + 1e-6)) if lookalike_total > 0 else 0.0
            
            current_lr = scheduler.get_last_lr()[0]
            
            # Logging
            logger.info(
                f"Epoch {epoch:03d}/{args.epochs-1} | "
                f"LR: {current_lr:.2e} | "
                f"Train Loss: {avg_train_loss:.4f} | "
                f"Val Loss: {avg_val_loss:.4f} | "
                f"mIoU: {val_miou:.4f} | "
                f"Oil IoU: {oil_iou:.4f} | "
                f"Oil Dice: {mean_oil_dice:.4f} | "
                f"Lookalike->Oil: {lookalike_as_oil_rate:.4f}"
            )
            
            with open(csv_path, mode="a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([
                    epoch, avg_train_loss, avg_val_loss, val_miou, 
                    oil_iou, lookalike_iou, bg_iou, mean_oil_dice, 
                    lookalike_as_oil_rate, current_lr
                ])
                
            # Checkpointing
            checkpoint_state = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'val_oil_iou': oil_iou,
                'val_miou': val_miou,
                'val_loss': avg_val_loss,
                'class_weights': class_weights
            }
            
            # Save latest
            torch.save(checkpoint_state, checkpoint_dir / "latest.pt")
            
            # Save best oil_spill IoU
            if oil_iou > best_oil_iou:
                best_oil_iou = oil_iou
                torch.save(checkpoint_state, checkpoint_dir / "best_oil_iou.pt")
                logger.info(f"[*] New best oil IoU: {best_oil_iou:.4f} - Saved model.")
            
            scheduler.step()

    except KeyboardInterrupt:
        logger.warning("Training interrupted by user. Saving latest checkpoint...")
        checkpoint_state = {
            'epoch': epoch if 'epoch' in locals() else start_epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'val_oil_iou': oil_iou if 'oil_iou' in locals() else -1.0,
            'val_miou': val_miou if 'val_miou' in locals() else -1.0,
            'val_loss': avg_val_loss if 'avg_val_loss' in locals() else -1.0,
            'class_weights': class_weights
        }
        torch.save(checkpoint_state, checkpoint_dir / "latest.pt")
        logger.info("Saved latest.pt gracefully.")
        sys.exit(0)
        
    logger.info("Training complete.")

if __name__ == "__main__":
    main()
