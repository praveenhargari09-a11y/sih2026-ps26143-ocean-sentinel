"""
backend/detection/model.py
===========================
U-Net model for oil spill segmentation in SAR imagery.

Architecture:
  - Encoder : ResNet-50 pretrained on ImageNet (transfer learning)
  - Decoder : U-Net decoder with skip connections
  - Head    : 1×1 convolution → n_classes logits

Loss functions:
  - DiceLoss       : handles class imbalance (oil spills are very small)
  - CombinedLoss   : weighted sum of CrossEntropy + Dice (best of both)

Classes (default):
  0 = background ocean
  1 = oil spill
  2 = lookalike (biogenic slick, wind shadow, etc.)
"""

from __future__ import annotations

import os
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from loguru import logger

try:
    import segmentation_models_pytorch as smp
    SMP_AVAILABLE = True
except ImportError:
    SMP_AVAILABLE = False
    logger.warning(
        "segmentation_models_pytorch not installed. "
        "Using built-in fallback U-Net. "
        "Install with: pip install segmentation-models-pytorch"
    )


# ---------------------------------------------------------------------------
# Fallback minimal U-Net (no SMP dependency)
# ---------------------------------------------------------------------------

class _DoubleConv(nn.Module):
    """Two consecutive 3×3 conv + BN + ReLU blocks."""

    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class _FallbackUNet(nn.Module):
    """
    Lightweight pure-PyTorch U-Net fallback used when
    segmentation_models_pytorch is unavailable.

    Channels: 64 → 128 → 256 → 512
    """

    def __init__(self, in_channels: int = 2, num_classes: int = 3) -> None:
        super().__init__()
        # Encoder
        self.enc1 = _DoubleConv(in_channels, 64)
        self.enc2 = _DoubleConv(64, 128)
        self.enc3 = _DoubleConv(128, 256)
        self.enc4 = _DoubleConv(256, 512)
        self.pool = nn.MaxPool2d(2)
        # Bottleneck
        self.bottleneck = _DoubleConv(512, 1024)
        # Decoder
        self.up4 = nn.ConvTranspose2d(1024, 512, 2, stride=2)
        self.dec4 = _DoubleConv(1024, 512)
        self.up3 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.dec3 = _DoubleConv(512, 256)
        self.up2 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.dec2 = _DoubleConv(256, 128)
        self.up1 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.dec1 = _DoubleConv(128, 64)
        # Head
        self.head = nn.Conv2d(64, num_classes, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))
        b = self.bottleneck(self.pool(e4))

        d4 = self.dec4(torch.cat([self.up4(b), e4], dim=1))
        d3 = self.dec3(torch.cat([self.up3(d4), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))
        return self.head(d1)


# ---------------------------------------------------------------------------
# Main model class
# ---------------------------------------------------------------------------

class OilSpillUNet(nn.Module):
    """
    U-Net segmentation model for oil spill detection in dual-polarization SAR.

    When ``segmentation_models_pytorch`` (SMP) is available:
      - Encoder : ResNet-50 with ImageNet weights
      - Decoder : U-Net architecture from SMP

    Falls back to a smaller custom U-Net otherwise.

    Parameters
    ----------
    num_classes : int
        Number of output classes.  Default 3 (background / oil / lookalike).
    encoder : str
        SMP encoder name (e.g. 'resnet50', 'efficientnet-b4').
    encoder_weights : str
        Pretrained weights name ('imagenet' or None).
    in_channels : int
        Number of SAR input channels.  Default 2 (VV + VH polarizations).
    """

    def __init__(
        self,
        num_classes: int = 3,
        encoder: str = "resnet50",
        encoder_weights: str = "imagenet",
        in_channels: int = 2,
    ) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.in_channels = in_channels

        if SMP_AVAILABLE:
            self._model = smp.Unet(
                encoder_name=encoder,
                encoder_weights=encoder_weights,
                in_channels=in_channels,
                classes=num_classes,
                activation=None,  # raw logits — loss functions apply softmax internally
            )
            logger.info(
                f"OilSpillUNet: SMP Unet, encoder={encoder}, "
                f"in_channels={in_channels}, classes={num_classes}"
            )
        else:
            self._model = _FallbackUNet(in_channels=in_channels, num_classes=num_classes)
            logger.warning("OilSpillUNet: Using fallback lightweight U-Net (SMP unavailable)")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Input SAR tensor, shape (B, C, H, W).

        Returns
        -------
        torch.Tensor
            Logits, shape (B, num_classes, H, W).
        """
        return self._model(x)

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with softmax applied.

        Returns
        -------
        torch.Tensor
            Class probabilities, shape (B, num_classes, H, W), values in [0,1].
        """
        return F.softmax(self.forward(x), dim=1)

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """
        Return argmax class map.

        Returns
        -------
        torch.Tensor
            Class indices, shape (B, H, W), dtype int64.
        """
        return torch.argmax(self.predict_proba(x), dim=1)


# ---------------------------------------------------------------------------
# Loss functions
# ---------------------------------------------------------------------------

class DiceLoss(nn.Module):
    """
    Soft Dice Loss for multi-class segmentation.

    Dice = 2·|P∩T| / (|P| + |T|)

    For multi-class, Dice is computed per-class then averaged
    (macro Dice). A smoothing constant ε prevents division by zero.

    Parameters
    ----------
    smooth : float
        Laplace smoothing constant.
    ignore_index : int, optional
        Class index to ignore (e.g. void label = 255).
    """

    def __init__(self, smooth: float = 1.0, ignore_index: int = -1) -> None:
        super().__init__()
        self.smooth = smooth
        self.ignore_index = ignore_index

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Compute Dice loss.

        Parameters
        ----------
        logits : torch.Tensor, shape (B, C, H, W)
        targets : torch.Tensor, shape (B, H, W), dtype int64

        Returns
        -------
        torch.Tensor
            Scalar Dice loss.
        """
        num_classes = logits.shape[1]
        probs = F.softmax(logits, dim=1)  # (B, C, H, W)

        # One-hot encode targets → (B, C, H, W)
        targets_one_hot = F.one_hot(targets.clamp(0, num_classes - 1), num_classes)
        targets_one_hot = targets_one_hot.permute(0, 3, 1, 2).float()

        # Create mask for valid (non-ignored) pixels
        if self.ignore_index >= 0:
            valid = (targets != self.ignore_index).unsqueeze(1).float()
            probs = probs * valid
            targets_one_hot = targets_one_hot * valid

        # Flatten spatial dimensions
        probs_flat = probs.view(probs.shape[0], num_classes, -1)          # (B, C, N)
        targets_flat = targets_one_hot.view(targets_one_hot.shape[0], num_classes, -1)  # (B, C, N)

        # Per-class Dice
        intersection = (probs_flat * targets_flat).sum(dim=2)             # (B, C)
        cardinality = probs_flat.sum(dim=2) + targets_flat.sum(dim=2)    # (B, C)
        dice_score = (2.0 * intersection + self.smooth) / (cardinality + self.smooth)

        # Mean over classes and batch
        dice_loss = 1.0 - dice_score.mean()
        return dice_loss


class CombinedLoss(nn.Module):
    """
    Weighted combination of Cross-Entropy and Dice Loss.

    Loss = α · CrossEntropy + β · Dice

    Cross-Entropy handles confident per-pixel classification;
    Dice handles class imbalance inherent in spill detection
    (oil covers < 5% of typical SAR scenes).

    Parameters
    ----------
    ce_weight : float
        Weight α for CrossEntropy component.
    dice_weight : float
        Weight β for Dice component.
    class_weights : torch.Tensor, optional
        Per-class weights for CrossEntropy (shape (num_classes,)).
    smooth : float
        Smoothing for DiceLoss.
    ignore_index : int
        Ignore index for CrossEntropy.
    """

    def __init__(
        self,
        ce_weight: float = 0.5,
        dice_weight: float = 0.5,
        class_weights: Optional[torch.Tensor] = None,
        smooth: float = 1.0,
        ignore_index: int = 255,
    ) -> None:
        super().__init__()
        self.ce_weight = ce_weight
        self.dice_weight = dice_weight

        self.ce_loss = nn.CrossEntropyLoss(
            weight=class_weights,
            ignore_index=ignore_index,
        )
        self.dice_loss = DiceLoss(smooth=smooth)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Compute combined loss.

        Parameters
        ----------
        logits : torch.Tensor, shape (B, C, H, W)
        targets : torch.Tensor, shape (B, H, W), dtype int64

        Returns
        -------
        torch.Tensor
            Scalar combined loss.
        """
        ce = self.ce_loss(logits, targets)
        dice = self.dice_loss(logits, targets)
        return self.ce_weight * ce + self.dice_weight * dice


# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------

def get_model(
    num_classes: int = 3,
    encoder: str = "resnet50",
    in_channels: int = 2,
) -> OilSpillUNet:
    """
    Instantiate a fresh OilSpillUNet.

    Parameters
    ----------
    num_classes : int
    encoder : str
        SMP encoder backbone name.
    in_channels : int

    Returns
    -------
    OilSpillUNet
    """
    return OilSpillUNet(
        num_classes=num_classes,
        encoder=encoder,
        encoder_weights="imagenet",
        in_channels=in_channels,
    )


def load_model(
    checkpoint_path: str,
    device: str = "cpu",
    num_classes: int = 3,
    encoder: str = "resnet50",
    in_channels: int = 2,
) -> OilSpillUNet:
    """
    Load a trained OilSpillUNet from a checkpoint file.

    The checkpoint must be a dict with at least the key ``model_state_dict``,
    or directly the model state dict.

    Parameters
    ----------
    checkpoint_path : str
        Path to the ``.pt`` checkpoint file.
    device : str
        Target device ('cpu', 'cuda', 'cuda:0', etc.)
    num_classes : int
    encoder : str
    in_channels : int

    Returns
    -------
    OilSpillUNet
        Model in eval mode on the specified device.

    Raises
    ------
    FileNotFoundError
        If the checkpoint file does not exist.
    """
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    model = OilSpillUNet(
        num_classes=num_classes,
        encoder=encoder,
        in_channels=in_channels,
    )

    state = torch.load(checkpoint_path, map_location=device)

    # Support both raw state dict and wrapped checkpoint
    if isinstance(state, dict) and "model_state_dict" in state:
        model.load_state_dict(state["model_state_dict"])
        epoch = state.get("epoch", "?")
        val_iou = state.get("val_iou", "?")
        logger.info(f"Loaded checkpoint (epoch={epoch}, val_iou={val_iou}) from {checkpoint_path}")
    else:
        model.load_state_dict(state)
        logger.info(f"Loaded state dict from {checkpoint_path}")

    model.to(device)
    model.eval()
    return model


# ---------------------------------------------------------------------------
# Quick self-test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    logger.info("Testing OilSpillUNet …")
    model = get_model(num_classes=3, encoder="resnet50", in_channels=2)
    dummy = torch.randn(2, 2, 512, 512)  # batch=2, channels=2, 512×512
    with torch.no_grad():
        logits = model(dummy)
    logger.info(f"Output logits shape: {logits.shape}")  # (2, 3, 512, 512)

    criterion = CombinedLoss(ce_weight=0.5, dice_weight=0.5)
    targets = torch.randint(0, 3, (2, 512, 512))
    loss = criterion(logits, targets)
    logger.info(f"Combined loss: {loss.item():.4f}")
    logger.info("Model self-test passed ✓")
