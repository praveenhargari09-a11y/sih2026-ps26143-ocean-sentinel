"""
backend/detection/inference.py
================================
Production inference engine for oil spill detection in SAR imagery.

Pipeline:
    SAR GeoTIFF → preprocess → tile → model inference → merge →
    argmax mask → polygonize → geometry properties → GeoJSON output

Returns a structured dict with:
  - spill_polygons  : list of GeoJSON Features (class=1)
  - lookalike_polygons: list of GeoJSON Features (class=2)
  - spill_mask      : numpy array of argmax class predictions
  - metadata        : CRS, transform, acquisition time
"""

from __future__ import annotations

import os
from typing import Optional

import numpy as np
import torch
import rasterio
import rasterio.features
from rasterio.transform import Affine
from shapely.geometry import shape, mapping
from shapely.ops import unary_union
from loguru import logger

from .preprocess import SARPreprocessor
from .model import OilSpillUNet, load_model, get_model
from .age_estimator import SpillAgeEstimator


# Approximate metres per degree at mid-latitudes
M_PER_DEG_LAT = 111_320.0  # metres per degree latitude
M_PER_DEG_LON = 111_320.0  # rough approximation (accurate at equator)


class SpillDetector:
    """
    End-to-end oil spill detector for SAR GeoTIFF scenes.

    Parameters
    ----------
    model_path : str, optional
        Path to a trained ``.pt`` checkpoint.  If None (or not found), the
        model is instantiated with random weights (for testing/demo).
    device : str
        Torch compute device ('cpu', 'cuda', etc.).
    tile_size : int
        Tile size used for inference (must match training tile size).
    overlap : int
        Overlap pixels between adjacent tiles.
    num_classes : int
        Number of segmentation classes.
    min_area_pixels : int
        Minimum polygon area (pixels²) to keep after polygonization.
        Filters noise and tiny false positives.
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        device: str = "cpu",
        tile_size: int = 512,
        overlap: int = 64,
        num_classes: int = 3,
        min_area_pixels: int = 100,
    ) -> None:
        self.device = device
        self.tile_size = tile_size
        self.overlap = overlap
        self.num_classes = num_classes
        self.min_area_pixels = min_area_pixels

        self.preprocessor = SARPreprocessor()
        self.age_estimator = SpillAgeEstimator()

        # Load model
        if model_path and os.path.exists(model_path):
            logger.info(f"Loading checkpoint from {model_path}")
            self.model = load_model(model_path, device=device, num_classes=num_classes)
        else:
            logger.warning(
                f"Model checkpoint not found at {model_path!r}. "
                "Using untrained model — predictions will be random."
            )
            self.model = get_model(num_classes=num_classes)
            self.model.to(device)
            self.model.eval()

    # ------------------------------------------------------------------
    # Main public API
    # ------------------------------------------------------------------

    def detect(self, sar_path: str) -> dict:
        """
        Run the full detection pipeline on a SAR GeoTIFF.

        Parameters
        ----------
        sar_path : str
            Path to the SAR GeoTIFF (one or two bands: VV [+ VH]).

        Returns
        -------
        dict with keys:
          - ``spill_polygons``    : list[dict] — GeoJSON Features for oil spill class
          - ``lookalike_polygons``: list[dict] — GeoJSON Features for lookalike class
          - ``spill_mask``        : np.ndarray (H, W), dtype int32, class indices
          - ``metadata``          : dict with crs, transform, acquisition_time
          - ``statistics``        : dict with total_spill_area_km2, n_spill_polygons, etc.
        """
        logger.info(f"Running spill detection on: {sar_path}")

        # ── 1. Load scene ─────────────────────────────────────────────
        arr, meta = self.preprocessor.load_scene(sar_path)
        H, W = arr.shape[:2]
        logger.debug(f"Scene shape: {arr.shape}, CRS: {meta['crs']}")

        # ── 2. Preprocess ─────────────────────────────────────────────
        arr = self.preprocessor.calibrate_sigma0(arr)
        arr = self.preprocessor.apply_speckle_filter(arr, window=5)
        arr = self.preprocessor.normalize(arr)

        # Ensure at least 2 channels (duplicate if single-pol)
        if arr.ndim == 2 or arr.shape[2] == 1:
            arr = np.concatenate([arr] * 2, axis=-1) if arr.ndim == 3 else \
                  np.stack([arr, arr], axis=-1)

        # ── 3. Tile ───────────────────────────────────────────────────
        tiles = self.preprocessor.tile(arr, tile_size=self.tile_size, overlap=self.overlap)

        # ── 4. Batch inference ────────────────────────────────────────
        self.model.eval()
        with torch.no_grad():
            for tile_dict in tiles:
                patch = tile_dict["tile"]  # (H_t, W_t, C)
                # Transpose → (C, H_t, W_t), add batch dim → (1, C, H_t, W_t)
                tensor = torch.from_numpy(
                    patch.transpose(2, 0, 1)[np.newaxis]
                ).float().to(self.device)

                logits = self.model(tensor)  # (1, n_classes, H_t, W_t)
                probs = torch.softmax(logits, dim=1)
                tile_dict["prediction"] = probs[0].cpu().numpy()  # (n_classes, H_t, W_t)

        # ── 5. Merge tiles ────────────────────────────────────────────
        prob_map = self.preprocessor.merge_tiles(
            tiles, original_shape=(H, W), n_classes=self.num_classes
        )  # (n_classes, H, W)

        # Argmax class map
        class_map = np.argmax(prob_map, axis=0).astype(np.int32)  # (H, W)

        # ── 6. Polygonize ─────────────────────────────────────────────
        transform = meta["transform"]
        spill_polygons = self._polygonize(class_map, class_id=1, transform=transform, crs=meta["crs"])
        lookalike_polygons = self._polygonize(class_map, class_id=2, transform=transform, crs=meta["crs"])

        # ── 7. Geometry stats ─────────────────────────────────────────
        total_spill_area_km2 = sum(p["properties"].get("area_km2", 0) for p in spill_polygons)
        total_lookalike_area_km2 = sum(p["properties"].get("area_km2", 0) for p in lookalike_polygons)

        statistics = {
            "total_spill_area_km2": round(total_spill_area_km2, 4),
            "n_spill_polygons": len(spill_polygons),
            "total_lookalike_area_km2": round(total_lookalike_area_km2, 4),
            "n_lookalike_polygons": len(lookalike_polygons),
            "scene_width": W,
            "scene_height": H,
        }

        result = {
            "spill_polygons": spill_polygons,
            "lookalike_polygons": lookalike_polygons,
            "spill_mask": class_map,
            "prob_map": prob_map,
            "metadata": meta,
            "statistics": statistics,
        }
        logger.info(
            f"Detection complete: {len(spill_polygons)} spill polygons "
            f"({total_spill_area_km2:.2f} km²), "
            f"{len(lookalike_polygons)} lookalike polygons"
        )
        return result

    # ------------------------------------------------------------------
    # Polygonization helper
    # ------------------------------------------------------------------

    def _polygonize(
        self,
        class_map: np.ndarray,
        class_id: int,
        transform: Affine,
        crs,
    ) -> list[dict]:
        """
        Extract GeoJSON polygons for a given class from the class map.

        Uses ``rasterio.features.shapes`` to trace connected pixel regions.
        Adds geometry properties: area_km2, perimeter_km, centroid_lon/lat,
        elongation_ratio (major/minor axis lengths from bounding box).

        Parameters
        ----------
        class_map : np.ndarray (H, W), int32
        class_id : int
            Class to polygonize.
        transform : rasterio.transform.Affine
        crs : rasterio CRS object

        Returns
        -------
        list[dict]
            List of GeoJSON Feature dicts with properties.
        """
        # Binary mask for this class
        binary = (class_map == class_id).astype(np.uint8)

        if binary.sum() == 0:
            return []

        features = []
        # rasterio.features.shapes yields (geometry, value) pairs
        for geom_dict, val in rasterio.features.shapes(binary, transform=transform):
            if val != 1:
                continue

            poly = shape(geom_dict)
            if not poly.is_valid:
                poly = poly.buffer(0)

            # Area in km² (degrees → rough km²)
            # We use bounding box pixel count for a more accurate estimate
            area_deg2 = poly.area
            # Approximate conversion: 1 deg² ≈ 111.32² km² at equator
            area_km2 = area_deg2 * (111.32 ** 2)

            # Filter tiny polygons
            pixel_area = binary.sum()  # rough guard
            if area_km2 < (self.min_area_pixels * 1e-4):  # very rough threshold
                continue

            # Perimeter in km
            perimeter_deg = poly.length
            perimeter_km = perimeter_deg * 111.32

            # Centroid
            centroid = poly.centroid
            centroid_lon = centroid.x
            centroid_lat = centroid.y

            # Elongation ratio from bounding box
            minx, miny, maxx, maxy = poly.bounds
            bbox_w = (maxx - minx) * 111.32  # km
            bbox_h = (maxy - miny) * 111.32  # km
            elongation = max(bbox_w, bbox_h) / (min(bbox_w, bbox_h) + 1e-6)

            feature = {
                "type": "Feature",
                "geometry": mapping(poly),
                "properties": {
                    "class_id": class_id,
                    "class_name": {1: "oil_spill", 2: "lookalike"}.get(class_id, "unknown"),
                    "area_km2": round(area_km2, 4),
                    "perimeter_km": round(perimeter_km, 4),
                    "centroid_lon": round(centroid_lon, 6),
                    "centroid_lat": round(centroid_lat, 6),
                    "elongation_ratio": round(elongation, 3),
                },
            }
            features.append(feature)

        logger.debug(f"Polygonized {len(features)} features for class {class_id}")
        return features


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import sys, json
    from pathlib import Path

    sar_path = sys.argv[1] if len(sys.argv) > 1 else None
    model_path = sys.argv[2] if len(sys.argv) > 2 else os.getenv("MODEL_CHECKPOINT", "")

    if not sar_path:
        logger.error("Usage: python -m backend.detection.inference <sar.tif> [model.pt]")
        sys.exit(1)

    detector = SpillDetector(model_path=model_path or None)
    result = detector.detect(sar_path)

    print(json.dumps(result["statistics"], indent=2))
    print(f"Spill polygons: {len(result['spill_polygons'])}")
    print(f"Lookalike polygons: {len(result['lookalike_polygons'])}")
