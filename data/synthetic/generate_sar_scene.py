"""
Generate a synthetic Sentinel-1 SAR-like GeoTIFF for oil spill demonstration.
Outputs: a 2048x2048 float32 VV/VH dual-polarization GeoTIFF + binary mask TIFF.
"""
import argparse, math, os
from pathlib import Path
import numpy as np
from loguru import logger

try:
    import rasterio
    from rasterio.transform import from_bounds
    from rasterio.crs import CRS
    HAS_RASTERIO = True
except ImportError:
    HAS_RASTERIO = False
    logger.warning('rasterio not installed; saving as numpy .npy instead')

class SyntheticSARGenerator:
    def generate(self, config_path: str, output_dir: str) -> dict:
        import yaml
        with open(config_path) as f:
            cfg = yaml.safe_load(f)
        spill = cfg['spill']
        
        lon_c = spill['centroid_lon']
        lat_c = spill['centroid_lat']
        major_km = spill.get('major_axis_km', 8.0)
        minor_km = spill.get('minor_axis_km', 2.5)
        angle_deg = spill.get('orientation_degrees', 45)
        H, W = 2048, 2048
        
        # Scene extent: ±0.5 degrees around centroid
        extent_deg = 0.5
        lon_min, lon_max = lon_c - extent_deg, lon_c + extent_deg
        lat_min, lat_max = lat_c - extent_deg, lat_c + extent_deg
        
        # --- Generate VV channel (backscatter in dB) ---
        rng = np.random.default_rng(42)
        # Background ocean: -15 dB mean, 2 dB std (Gaussian speckle)
        vv = rng.normal(-15.0, 2.0, (H, W)).astype(np.float32)
        vh = rng.normal(-22.0, 2.5, (H, W)).astype(np.float32)  # cross-pol
        mask = np.zeros((H, W), dtype=np.uint8)  # 0=background
        
        # Convert spill geometry to pixel coordinates
        def lon_to_col(lon): return int((lon - lon_min) / (lon_max - lon_min) * W)
        def lat_to_row(lat): return int((lat_max - lat) / (lat_max - lat_min) * H)
        
        cx, cy = W // 2, H // 2  # pixel center = scene center
        
        # Convert km to pixels
        km_per_deg_lon = 111.32 * math.cos(math.radians(lat_c))
        km_per_deg_lat = 110.574
        px_per_km_x = W / ((lon_max - lon_min) * km_per_deg_lon)
        px_per_km_y = H / ((lat_max - lat_min) * km_per_deg_lat)
        a_px = int(major_km / 2 * px_per_km_x)  # semi-major in pixels
        b_px = int(minor_km / 2 * px_per_km_y)  # semi-minor
        angle_rad = math.radians(angle_deg)
        
        # Draw oil spill ellipse
        ys, xs = np.mgrid[0:H, 0:W]
        dx = xs - cx
        dy = ys - cy
        # Rotate
        dx_r = dx * math.cos(angle_rad) + dy * math.sin(angle_rad)
        dy_r = -dx * math.sin(angle_rad) + dy * math.cos(angle_rad)
        ellipse_mask = (dx_r / max(a_px, 1))**2 + (dy_r / max(b_px, 1))**2 <= 1
        
        # Oil spill: lower backscatter -21 dB ± 1.5 dB
        vv[ellipse_mask] = rng.normal(-21.0, 1.5, ellipse_mask.sum()).astype(np.float32)
        vh[ellipse_mask] = rng.normal(-28.0, 1.5, ellipse_mask.sum()).astype(np.float32)
        mask[ellipse_mask] = 1
        
        # Add 1-2 lookalike patches
        for lk_idx in range(2):
            lk_cx = cx + rng.integers(-400, 400)
            lk_cy = cy + rng.integers(-400, 400)
            lk_a = int(a_px * rng.uniform(0.3, 0.7))
            lk_b = int(b_px * rng.uniform(0.3, 0.7))
            lk_angle = rng.uniform(0, math.pi)
            lk_dx = xs - lk_cx
            lk_dy = ys - lk_cy
            lk_dxr = lk_dx * math.cos(lk_angle) + lk_dy * math.sin(lk_angle)
            lk_dyr = -lk_dx * math.sin(lk_angle) + lk_dy * math.cos(lk_angle)
            lk_mask = (lk_dxr / max(lk_a, 1))**2 + (lk_dyr / max(lk_b, 1))**2 <= 1
            lk_mask &= ~ellipse_mask  # don't overlap oil spill
            vv[lk_mask] = rng.normal(-18.5, 1.0, lk_mask.sum()).astype(np.float32)
            mask[lk_mask] = 2  # lookalike class
        
        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        sar_path = str(out_dir / 'synthetic_sar.tif')
        mask_path = str(out_dir / 'synthetic_mask.tif')
        
        if HAS_RASTERIO:
            transform = from_bounds(lon_min, lat_min, lon_max, lat_max, W, H)
            crs = CRS.from_epsg(4326)
            with rasterio.open(
                sar_path, 'w', driver='GTiff', height=H, width=W, count=2,
                dtype='float32', crs=crs, transform=transform
            ) as dst:
                dst.write(vv, 1)
                dst.write(vh, 2)
                dst.update_tags(acquisition_time=spill.get('acquisition_time', '2024-06-15T08:30:00Z'))
            
            with rasterio.open(
                mask_path, 'w', driver='GTiff', height=H, width=W, count=1,
                dtype='uint8', crs=crs, transform=transform
            ) as dst:
                dst.write(mask, 1)
            logger.info(f'Saved synthetic SAR: {sar_path}')
            logger.info(f'Saved synthetic mask: {mask_path}')
        else:
            sar_path = str(out_dir / 'synthetic_sar.npy')
            mask_path = str(out_dir / 'synthetic_mask.npy')
            np.save(sar_path, np.stack([vv, vh], axis=0))
            np.save(mask_path, mask)
        
        return {'sar_path': sar_path, 'mask_path': mask_path, 'height': H, 'width': W,
                'centroid_lon': lon_c, 'centroid_lat': lat_c,
                'lon_min': lon_min, 'lon_max': lon_max, 'lat_min': lat_min, 'lat_max': lat_max}

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate synthetic SAR scene')
    parser.add_argument('--config', default='data/synthetic/scenario_config.yaml')
    parser.add_argument('--output', default='data/raw')
    args = parser.parse_args()
    gen = SyntheticSARGenerator()
    result = gen.generate(args.config, args.output)
    print(result)
