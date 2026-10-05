"""
backend/detection/preprocess.py
================================
SAR (Synthetic Aperture Radar) Scene Preprocessor.

Handles the full preprocessing pipeline from raw GeoTIFF to normalized,
tiled arrays ready for deep-learning inference.

Pipeline:
    load_scene → calibrate_sigma0 → apply_speckle_filter → normalize → tile
    (reassemble with) merge_tiles
"""

from __future__ import annotations

import os
import math
from typing import Optional

import numpy as np
import rasterio
from rasterio.transform import Affine
from loguru import logger
from scipy.ndimage import uniform_filter


class SARPreprocessor:
    """
    End-to-end SAR preprocessing pipeline.

    Attributes
    ----------
    calibration_offset : float
        Radiometric calibration constant (dB).  Default matches Sentinel-1
        IW GRDH convention (83.0 dB).
    """

    DEFAULT_CALIBRATION_OFFSET: float = 83.0  # Sentinel-1 calibration constant

    def __init__(self, calibration_offset: float = DEFAULT_CALIBRATION_OFFSET) -> None:
        self.calibration_offset = calibration_offset

    # ------------------------------------------------------------------
    # I/O
    # ------------------------------------------------------------------

    def load_scene(self, path: str) -> tuple[np.ndarray, dict]:
        """
        Load a GeoTIFF SAR scene from disk.

        Parameters
        ----------
        path : str
            Absolute or relative path to the GeoTIFF file.

        Returns
        -------
        arr : np.ndarray
            Float32 array of shape (H, W, C) where C = number of bands.
        meta : dict
            Rasterio metadata dict containing ``crs``, ``transform``,
            ``width``, ``height``, ``count``, ``dtype`` and any extra tags
            found in the file (e.g. ``acquisition_time``).

        Raises
        ------
        FileNotFoundError
            If ``path`` does not exist.
        rasterio.errors.RasterioIOError
            If the file cannot be opened as a GeoTIFF.
        """
        if not os.path.exists(path):
            raise FileNotFoundError(f"SAR scene not found: {path}")

        logger.info(f"Loading SAR scene: {path}")

        with rasterio.open(path) as src:
            # Read all bands → (C, H, W) float32
            data = src.read().astype(np.float32)  # shape: (C, H, W)
            meta = {
                "crs": src.crs,
                "transform": src.transform,
                "width": src.width,
                "height": src.height,
                "count": src.count,
                "dtype": str(src.dtypes[0]),
                "nodata": src.nodata,
                "tags": src.tags(),
            }
            # Try to extract acquisition time from TIFF tags
            tags = src.tags()
            meta["acquisition_time"] = tags.get(
                "acquisition_time", tags.get("TIFFTAG_DATETIME", "")
            )

        # Transpose to (H, W, C)
        arr = np.transpose(data, (1, 2, 0))
        logger.debug(f"Scene loaded: shape={arr.shape}, dtype={arr.dtype}")
        return arr, meta

    # ------------------------------------------------------------------
    # Radiometric Calibration
    # ------------------------------------------------------------------

    def calibrate_sigma0(self, arr: np.ndarray) -> np.ndarray:
        """
        Convert raw Digital Numbers (DN) to Sigma-0 in decibels (dB).

        Formula:
            σ⁰ (dB) = 10 · log₁₀(DN²) − offset
                     = 20 · log₁₀(DN) − offset

        This matches the standard Sentinel-1 radiometric calibration formula.
        Values of DN ≤ 0 are replaced with a small epsilon before log to avoid
        -inf.

        Parameters
        ----------
        arr : np.ndarray
            Raw DN array, shape (H, W, C) or (H, W).

        Returns
        -------
        np.ndarray
            Sigma-0 array in dB, same shape as input.
        """
        eps = 1e-6
        arr_safe = np.where(arr > 0, arr, eps)
        sigma0_db = 20.0 * np.log10(arr_safe) - self.calibration_offset
        logger.debug(
            f"Sigma-0 calibrated: min={sigma0_db.min():.2f} dB, max={sigma0_db.max():.2f} dB"
        )
        return sigma0_db.astype(np.float32)

    # ------------------------------------------------------------------
    # Speckle Filtering
    # ------------------------------------------------------------------

    def apply_speckle_filter(self, arr: np.ndarray, window: int = 5) -> np.ndarray:
        """
        Apply a Lee speckle filter to reduce multiplicative SAR speckle noise.

        The Lee filter estimates the local mean and variance within a sliding
        window and computes a weighted average:
            filtered = mean + k · (arr − mean)
        where k = local_var / (local_var + noise_var).

        The global noise variance is estimated as (mean / ENL)²
        where ENL (Equivalent Number of Looks) ≈ window².

        Parameters
        ----------
        arr : np.ndarray
            Input array of shape (H, W, C) or (H, W).
        window : int
            Side length of the square filter window (pixels).  Must be odd.

        Returns
        -------
        np.ndarray
            Lee-filtered array, same shape and dtype as input.
        """
        if window % 2 == 0:
            window += 1  # Ensure odd window size

        # Work per-channel
        if arr.ndim == 2:
            return self._lee_filter_2d(arr, window)

        out = np.empty_like(arr)
        for c in range(arr.shape[2]):
            out[:, :, c] = self._lee_filter_2d(arr[:, :, c], window)
        logger.debug(f"Lee speckle filter applied: window={window}")
        return out

    @staticmethod
    def _lee_filter_2d(band: np.ndarray, window: int) -> np.ndarray:
        """
        Apply Lee filter to a single 2D band.

        Parameters
        ----------
        band : np.ndarray, shape (H, W)
        window : int

        Returns
        -------
        np.ndarray, shape (H, W)
        """
        # Local mean using uniform (box) filter
        local_mean = uniform_filter(band, size=window)

        # Local mean of squared values → needed for variance
        local_sq_mean = uniform_filter(band ** 2, size=window)

        # Local variance = E[X²] − E[X]²
        local_var = local_sq_mean - local_mean ** 2

        # Estimate global noise variance as 1/ENL of the overall variance
        enl = window ** 2  # Equivalent Number of Looks
        noise_var = np.var(band) / enl

        # Lee filter weight
        noise_var = max(noise_var, 1e-10)  # avoid division by zero
        k = local_var / (local_var + noise_var)

        # Apply filter
        filtered = local_mean + k * (band - local_mean)
        return filtered.astype(band.dtype)

    # ------------------------------------------------------------------
    # Normalization
    # ------------------------------------------------------------------

    def normalize(self, arr: np.ndarray,
                  clip_percentile: tuple[float, float] = (2.0, 98.0)) -> np.ndarray:
        """
        Clip to percentile range and min-max normalize to [0, 1].

        Percentile clipping removes extreme outliers (e.g., very bright ship
        reflections) that would otherwise dominate the normalization range.

        Parameters
        ----------
        arr : np.ndarray
            Input array, shape (H, W, C) or (H, W).
        clip_percentile : tuple[float, float]
            Lower and upper percentiles for clipping.

        Returns
        -------
        np.ndarray
            Normalized float32 array in [0, 1].
        """
        p_lo, p_hi = np.percentile(arr, clip_percentile[0]), np.percentile(arr, clip_percentile[1])
        clipped = np.clip(arr, p_lo, p_hi)
        denom = p_hi - p_lo
        if denom < 1e-10:
            logger.warning("Normalization: very small range detected, returning zeros.")
            return np.zeros_like(arr, dtype=np.float32)
        normalized = (clipped - p_lo) / denom
        logger.debug(f"Normalized: range [{p_lo:.4f}, {p_hi:.4f}] → [0, 1]")
        return normalized.astype(np.float32)

    # ------------------------------------------------------------------
    # Tiling
    # ------------------------------------------------------------------

    def tile(
        self,
        arr: np.ndarray,
        tile_size: int = 512,
        overlap: int = 64,
    ) -> list[dict]:
        """
        Tile a large SAR scene into overlapping patches for batch inference.

        The overlap region ensures that predictions at tile boundaries are
        covered by at least one full tile.  Use ``merge_tiles`` to reassemble.

        Parameters
        ----------
        arr : np.ndarray
            Input array of shape (H, W, C) or (H, W).
        tile_size : int
            Side length of each square tile in pixels.
        overlap : int
            Number of pixels of overlap between adjacent tiles.

        Returns
        -------
        list[dict]
            Each dict contains:
              - ``tile``: np.ndarray of shape (tile_size, tile_size, C)
              - ``row_start``, ``row_end``: pixel coordinates in the source image
              - ``col_start``, ``col_end``: pixel coordinates in the source image
        """
        if arr.ndim == 2:
            arr = arr[:, :, np.newaxis]  # add channel dim

        H, W, C = arr.shape
        stride = tile_size - overlap
        tiles: list[dict] = []

        row = 0
        while row < H:
            col = 0
            while col < W:
                row_end = min(row + tile_size, H)
                col_end = min(col + tile_size, W)
                row_start = row_end - tile_size  # may be negative at boundaries
                col_start = col_end - tile_size

                # Clamp to valid range
                row_start = max(0, row_start)
                col_start = max(0, col_start)

                patch = arr[row_start:row_end, col_start:col_end, :]

                # Pad if patch is smaller than tile_size
                pad_h = tile_size - patch.shape[0]
                pad_w = tile_size - patch.shape[1]
                if pad_h > 0 or pad_w > 0:
                    patch = np.pad(
                        patch,
                        ((0, pad_h), (0, pad_w), (0, 0)),
                        mode="reflect",
                    )

                tiles.append(
                    {
                        "tile": patch,
                        "row_start": row_start,
                        "row_end": row_end,
                        "col_start": col_start,
                        "col_end": col_end,
                    }
                )

                col += stride
                if col >= W and col_end < W:
                    break
                if col_end == W:
                    break

            row += stride
            if row >= H and row_end < H:
                break
            if row_end == H:
                break

        logger.debug(f"Tiled scene {(H, W, C)} into {len(tiles)} tiles of size {tile_size}x{tile_size}")
        return tiles

    # ------------------------------------------------------------------
    # Tile Merging
    # ------------------------------------------------------------------

    def merge_tiles(
        self,
        tiles: list[dict],
        original_shape: tuple[int, int],
        n_classes: int = 3,
    ) -> np.ndarray:
        """
        Reassemble per-tile predictions back into a full-scene prediction map.

        Uses *average blending* in overlap zones: each pixel accumulates the
        sum of all tile predictions that cover it, then divides by the count.
        This produces smooth transitions at tile boundaries.

        Parameters
        ----------
        tiles : list[dict]
            List of dicts with keys ``prediction`` (np.ndarray, shape
            (n_classes, tile_size, tile_size)), ``row_start``, ``row_end``,
            ``col_start``, ``col_end``.
        original_shape : tuple[int, int]
            ``(H, W)`` of the original scene.
        n_classes : int
            Number of segmentation classes.

        Returns
        -------
        np.ndarray
            Averaged prediction map, shape (n_classes, H, W), dtype float32.
        """
        H, W = original_shape
        accumulator = np.zeros((n_classes, H, W), dtype=np.float64)
        count_map = np.zeros((H, W), dtype=np.float64)

        for t in tiles:
            pred = t["prediction"]  # (n_classes, tile_H, tile_W)
            rs, re = t["row_start"], t["row_end"]
            cs, ce = t["col_start"], t["col_end"]

            tile_h = re - rs
            tile_w = ce - cs

            # Crop prediction to actual tile size (strip padding)
            pred_crop = pred[:, :tile_h, :tile_w]

            accumulator[:, rs:re, cs:ce] += pred_crop
            count_map[rs:re, cs:ce] += 1.0

        # Avoid division by zero (shouldn't happen with correct tiling)
        count_map = np.where(count_map == 0, 1.0, count_map)
        result = (accumulator / count_map[np.newaxis, :, :]).astype(np.float32)
        logger.debug(f"Merged {len(tiles)} tiles into prediction map {result.shape}")
        return result


# ---------------------------------------------------------------------------
# Quick self-test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import tempfile, rasterio
    from rasterio.transform import from_bounds

    logger.info("Running SARPreprocessor self-test …")
    pp = SARPreprocessor()

    # Create a tiny synthetic GeoTIFF for testing
    H, W = 128, 128
    data = np.random.randint(200, 800, (2, H, W), dtype=np.uint16)

    with tempfile.NamedTemporaryFile(suffix=".tif", delete=False) as f:
        tmp_path = f.name

    transform = from_bounds(60.0, 15.0, 65.0, 20.0, W, H)
    with rasterio.open(
        tmp_path, "w", driver="GTiff", height=H, width=W, count=2,
        dtype="uint16", crs="EPSG:4326", transform=transform,
    ) as dst:
        dst.write(data)

    arr, meta = pp.load_scene(tmp_path)
    logger.info(f"Loaded shape: {arr.shape}, CRS: {meta['crs']}")

    sigma0 = pp.calibrate_sigma0(arr)
    filtered = pp.apply_speckle_filter(sigma0, window=5)
    normalized = pp.normalize(filtered)
    tiles = pp.tile(normalized, tile_size=64, overlap=8)

    # Simulate predictions
    for t in tiles:
        t["prediction"] = np.random.rand(3, 64, 64).astype(np.float32)

    merged = pp.merge_tiles(tiles, original_shape=(H, W), n_classes=3)
    logger.info(f"Merged shape: {merged.shape}")
    logger.info("Self-test passed ✓")

    os.unlink(tmp_path)
