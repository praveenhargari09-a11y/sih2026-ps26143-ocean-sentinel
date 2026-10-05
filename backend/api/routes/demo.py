"""
demo.py — Convenience demo route that runs the full synthetic pipeline
in one API call without needing real SAR or AIS data.

GET /api/demo/generate  →  generates synthetic SAR scene + AIS tracks,
                            runs detection → drift → vessel ranking,
                            returns complete result bundle.
"""

import os
import sys
import uuid
import json
from pathlib import Path
from datetime import datetime, timezone

from fastapi import APIRouter, UploadFile
from loguru import logger

router = APIRouter(prefix="/demo", tags=["demo"])

# Base project root (two levels up from api/routes/)
PROJECT_ROOT = Path(__file__).resolve().parents[3]
SYNTH_DIR = PROJECT_ROOT / "data" / "synthetic"
SCENARIO_CONFIG = SYNTH_DIR / "scenario_config.yaml"


def _load_scenario_config():
    """Load scenario YAML config."""
    import yaml
    with open(SCENARIO_CONFIG, "r") as f:
        return yaml.safe_load(f)


def _build_mock_spill(config: dict, spill_id: str) -> dict:
    """Build a SpillRecord-like dict from the scenario config."""
    spill_cfg = config["spill"]
    return {
        "id": spill_id,
        "name": f"Demo Spill — {config['scenario']['name']}",
        "centroid_lon": spill_cfg["centroid_lon"],
        "centroid_lat": spill_cfg["centroid_lat"],
        "area_km2": spill_cfg["area_km2"],
        "acquisition_time": spill_cfg["acquisition_time"],
        "status": "analysed",
        "detection_time": datetime.now(timezone.utc).isoformat(),
        "spill_polygon": {
            "type": "Feature",
            "geometry": _ellipse_geojson(
                lon=spill_cfg["centroid_lon"],
                lat=spill_cfg["centroid_lat"],
                major_km=spill_cfg["major_axis_km"],
                minor_km=spill_cfg["minor_axis_km"],
                angle_deg=spill_cfg["orientation_degrees"],
            ),
            "properties": {
                "area_km2": spill_cfg["area_km2"],
                "perimeter_km": 2 * 3.14159 * ((spill_cfg["major_axis_km"] + spill_cfg["minor_axis_km"]) / 2),
                "elongation": spill_cfg["major_axis_km"] / spill_cfg["minor_axis_km"],
            },
        },
        "age_estimate": {
            "age_hours_min": 6.0,
            "age_hours_max": 18.0,
            "age_hours_mean": 12.0,
            "method": "fay_spreading",
        },
    }


def _ellipse_geojson(lon: float, lat: float, major_km: float, minor_km: float, angle_deg: float) -> dict:
    """Generate a GeoJSON Polygon approximating an ellipse."""
    import math

    n_points = 64
    # Degrees per km at given latitude
    km_per_deg_lon = 111.32 * math.cos(math.radians(lat))
    km_per_deg_lat = 110.574

    a_lon = major_km / km_per_deg_lon  # semi-major in degrees lon
    b_lat = minor_km / km_per_deg_lat  # semi-minor in degrees lat
    angle_rad = math.radians(angle_deg)

    coords = []
    for i in range(n_points + 1):
        theta = 2 * math.pi * i / n_points
        x = a_lon * math.cos(theta)
        y = b_lat * math.sin(theta)
        # Rotate by angle
        x_rot = x * math.cos(angle_rad) - y * math.sin(angle_rad)
        y_rot = x * math.sin(angle_rad) + y * math.cos(angle_rad)
        coords.append([lon + x_rot, lat + y_rot])

    return {"type": "Polygon", "coordinates": [coords]}


def _build_mock_drift(config: dict, spill_id: str) -> dict:
    """Build synthetic drift result from scenario config."""
    import math

    spill_cfg = config["spill"]
    drift_cfg = config["drift"]

    origin_lon = spill_cfg["centroid_lon"]
    origin_lat = spill_cfg["centroid_lat"]

    # Reverse drift direction for hindcast
    wind_rad = math.radians(drift_cfg["wind_direction_degrees"])
    current_rad = math.radians(drift_cfg["current_direction_degrees"])

    # Simple linear drift: combined wind + current
    drift_speed = drift_cfg["current_speed_ms"] * 3600 / 1000  # km/h
    wind_drift = drift_cfg["wind_speed_ms"] * 0.035 * 3600 / 1000  # 3.5% wind factor km/h

    hours = drift_cfg["hindcast_hours"]
    total_km = (drift_speed + wind_drift) * hours

    km_per_deg = 111.0
    delta_lon = (total_km * math.sin(current_rad)) / (km_per_deg * math.cos(math.radians(origin_lat)))
    delta_lat = (total_km * math.cos(current_rad)) / km_per_deg

    # Origin = spill centroid minus drift displacement
    origin_lon_est = origin_lon - delta_lon
    origin_lat_est = origin_lat - delta_lat

    # Build hindcast line (simplified: 6-hour steps backward)
    hindcast_coords = []
    for h in range(0, hours + 1, 6):
        frac = h / hours
        hindcast_coords.append([
            origin_lon - delta_lon * (1 - frac),
            origin_lat - delta_lat * (1 - frac),
        ])

    # Build forecast line (forward)
    forecast_hours = drift_cfg["forecast_hours"]
    forecast_total_km = (drift_speed + wind_drift) * forecast_hours
    fdelta_lon = (forecast_total_km * math.sin(wind_rad)) / (km_per_deg * math.cos(math.radians(origin_lat)))
    fdelta_lat = (forecast_total_km * math.cos(wind_rad)) / km_per_deg

    forecast_coords = []
    for h in range(0, forecast_hours + 1, 6):
        frac = h / forecast_hours
        forecast_coords.append([
            origin_lon + fdelta_lon * frac,
            origin_lat + fdelta_lat * frac,
        ])

    # Dynamically calculate origin time
    spill_cfg = config.get("spill", {})
    acq_time_str = spill_cfg.get("acquisition_time")
    origin_time_str = "2024-06-14T20:30:00Z"
    if acq_time_str:
        try:
            from datetime import datetime, timedelta
            acq_time = datetime.fromisoformat(acq_time_str.replace('Z', '+00:00'))
            mean_age = spill_cfg.get("age_estimate", {}).get("age_hours_mean", drift_cfg.get("hindcast_hours", 12.0))
            orig_dt = acq_time - timedelta(hours=mean_age)
            origin_time_str = orig_dt.isoformat().replace('+00:00', 'Z')
        except Exception:
            pass

    return {
        "spill_id": spill_id,
        "origin_lon": round(origin_lon_est, 4),
        "origin_lat": round(origin_lat_est, 4),
        "origin_time": origin_time_str,
        "uncertainty_km": 15.0,
        "confidence": 0.78,
        "hindcast_geojson": {
            "type": "FeatureCollection",
            "features": [{
                "type": "Feature",
                "geometry": {"type": "LineString", "coordinates": hindcast_coords},
                "properties": {"type": "hindcast", "duration_hours": hours},
            }],
        },
        "forecast_geojson": {
            "type": "FeatureCollection",
            "features": [{
                "type": "Feature",
                "geometry": {"type": "LineString", "coordinates": forecast_coords},
                "properties": {"type": "forecast", "duration_hours": forecast_hours},
            }],
        },
    }


def _build_mock_vessels(config: dict, origin_lon: float, origin_lat: float) -> list:
    """Build synthetic vessel ranking from scenario config."""
    import math
    import random
    vessels_cfg = config["ais"]["vessels"]
    spill_cfg = config["spill"]
    spill_lon = spill_cfg["centroid_lon"]
    spill_lat = spill_cfg["centroid_lat"]

    ranked = []
    for i, v in enumerate(vessels_cfg):
        is_guilty = v.get("guilty", False)

        # Proximity: guilty vessel gets ~5km, others 20-80km
        if is_guilty:
            proximity_km = round(random.uniform(3.0, 8.0), 1)
            prox_score = 0.95
            traj_score = 0.90
            darkness_score = 0.95
            speed_score = 0.75
        else:
            proximity_km = round(random.uniform(20.0, 80.0), 1)
            prox_score = round(max(0, 1 - proximity_km / 200), 2)
            traj_score = round(random.uniform(0.05, 0.40), 2)
            darkness_score = round(random.uniform(0.0, 0.15), 2)
            speed_score = round(random.uniform(0.0, 0.20), 2)

        type_scores = {
            "Tanker": 1.0, "Bulk Carrier": 0.8, "Cargo": 0.6,
            "General Cargo": 0.55, "Container": 0.5,
        }
        type_score = type_scores.get(v.get("vessel_type", "Cargo"), 0.5)

        total = round(
            0.35 * prox_score
            + 0.25 * traj_score
            + 0.20 * darkness_score
            + 0.10 * type_score
            + 0.10 * speed_score,
            3,
        )

        # Generate realistic multi-point track near the spill area
        n_points = 20
        heading_rad = math.radians(random.uniform(0, 360))
        speed_deg_step = 0.02  # ~2km per step

        if is_guilty:
            # Guilty vessel passes through origin, heading toward spill
            mid = n_points // 2
            track_coords = []
            for j in range(n_points):
                frac = (j - mid) / n_points
                lon = origin_lon + math.sin(heading_rad) * speed_deg_step * (j - mid)
                lat = origin_lat + math.cos(heading_rad) * speed_deg_step * (j - mid)
                # Add slight wobble for realism
                lon += random.gauss(0, 0.003)
                lat += random.gauss(0, 0.003)
                track_coords.append([round(lon, 5), round(lat, 5)])
        else:
            # Innocent vessels: start near the spill area, travel in a random direction
            offset_deg = proximity_km / 111.0
            start_lon = spill_lon + random.uniform(-offset_deg, offset_deg) * 0.3
            start_lat = spill_lat + random.uniform(-offset_deg, offset_deg) * 0.3
            track_coords = []
            for j in range(n_points):
                lon = start_lon + math.sin(heading_rad) * speed_deg_step * j
                lat = start_lat + math.cos(heading_rad) * speed_deg_step * j
                lon += random.gauss(0, 0.002)
                lat += random.gauss(0, 0.002)
                track_coords.append([round(lon, 5), round(lat, 5)])

        ais_gaps = []
        if is_guilty and v.get("ais_gap_start"):
            ais_gaps.append({
                "start": v["ais_gap_start"],
                "duration_hours": v.get("ais_gap_hours", 6),
                "lon_before": origin_lon - 0.02,
                "lat_before": origin_lat - 0.01,
            })

        ranked.append({
            "mmsi": v["mmsi"],
            "vessel_name": v["name"],
            "vessel_type": v.get("vessel_type", "Unknown"),
            "flag": v.get("flag", "Unknown"),
            "total_score": total,
            "score_breakdown": {
                "proximity": prox_score,
                "trajectory": traj_score,
                "darkness": darkness_score,
                "vessel_type": type_score,
                "speed_anomaly": speed_score,
            },
            "proximity_km": proximity_km,
            "ais_gaps": ais_gaps,
            "track_geojson": {
                "type": "Feature",
                "geometry": {"type": "LineString", "coordinates": track_coords},
                "properties": {"mmsi": v["mmsi"], "vessel_name": v["name"]},
            },
        })

    # Sort by score descending and assign ranks
    ranked.sort(key=lambda x: x["total_score"], reverse=True)
    for idx, v in enumerate(ranked):
        v["rank"] = idx + 1

    return ranked


@router.get("/generate", summary="Generate and run full synthetic demo pipeline")
async def generate_demo():
    """
    Generates synthetic SAR scene + AIS tracks from scenario_config.yaml,
    runs the full detection → drift → vessel attribution pipeline,
    and returns a complete result bundle for the dashboard.

    No real satellite data or AIS feeds required.
    """
    try:
        config = _load_scenario_config()
    except FileNotFoundError:
        logger.warning("scenario_config.yaml not found, using hardcoded defaults")
        config = {
            "scenario": {"name": "Arabian Sea Demo"},
            "spill": {
                "centroid_lon": 68.5,
                "centroid_lat": 20.3,
                "area_km2": 12.5,
                "major_axis_km": 8.0,
                "minor_axis_km": 2.5,
                "orientation_degrees": 45,
                "acquisition_time": "2024-06-15T08:30:00Z",
            },
            "drift": {
                "wind_speed_ms": 5.2,
                "wind_direction_degrees": 225,
                "current_speed_ms": 0.3,
                "current_direction_degrees": 180,
                "hindcast_hours": 48,
                "forecast_hours": 72,
            },
            "ais": {
                "vessels": [
                    {"mmsi": "123456789", "name": "MV OCEAN STAR", "vessel_type": "Tanker",
                     "flag": "Panama", "guilty": True, "ais_gap_start": "2024-06-14T22:00:00Z", "ais_gap_hours": 6},
                    {"mmsi": "234567890", "name": "MV BLUE WAVE", "vessel_type": "Cargo", "flag": "Liberia"},
                    {"mmsi": "345678901", "name": "MV PACIFIC CROSS", "vessel_type": "Bulk Carrier", "flag": "Marshall Islands"},
                    {"mmsi": "456789012", "name": "MV KERALA TRADER", "vessel_type": "General Cargo", "flag": "India"},
                    {"mmsi": "567890123", "name": "MV ARABIAN QUEEN", "vessel_type": "Tanker", "flag": "UAE"},
                ],
            },
        }

    spill_id = str(uuid.uuid4())[:8]
    spill = _build_mock_spill(config, spill_id)
    drift = _build_mock_drift(config, spill_id)
    vessels = _build_mock_vessels(config, drift["origin_lon"], drift["origin_lat"])

    logger.info(f"Generated demo scenario: spill_id={spill_id}, vessels={len(vessels)}")

    return {
        "message": f"Demo scenario '{config['scenario']['name']}' generated successfully",
        "spill_id": spill_id,
        "spill": spill,
        "drift": drift,
        "vessels": vessels,
        "scenario_name": config["scenario"]["name"],
    }


@router.post("/detect", summary="Upload SAR file and run detection pipeline")
async def detect_from_upload(file: UploadFile):
    """
    Accepts a SAR GeoTIFF upload, runs mock detection,
    and returns a full spill + drift + vessel result.
    """
    from fastapi import UploadFile as _UploadFile
    import math, random, tempfile, shutil

    random.seed()
    spill_id = str(uuid.uuid4())[:8]

    # Save uploaded file temporarily
    upload_dir = PROJECT_ROOT / "data" / "uploads"
    upload_dir.mkdir(parents=True, exist_ok=True)
    dest = upload_dir / f"{spill_id}_{file.filename}"
    with open(dest, "wb") as f:
        shutil.copyfileobj(file.file, f)
    logger.info(f"Saved uploaded SAR file to {dest}")

    # Try to read actual image metadata
    centroid_lon, centroid_lat = 68.5, 20.3  # defaults
    area_km2 = 12.5
    try:
        import rasterio
        with rasterio.open(dest) as src:
            bounds = src.bounds
            if src.crs and bounds.left != 0:
                centroid_lon = (bounds.left + bounds.right) / 2
                centroid_lat = (bounds.top + bounds.bottom) / 2
                # Estimate area from pixel count with foreground
                data = src.read(1)
                import numpy as np
                width_km = abs(bounds.right - bounds.left) * 111 * math.cos(math.radians(centroid_lat))
                height_km = abs(bounds.top - bounds.bottom) * 111
                area_km2 = round(width_km * height_km * 0.15, 2)  # assume ~15% is spill
            logger.info(f"SAR metadata: bounds={bounds}, crs={src.crs}, bands={src.count}")
    except Exception as e:
        logger.warning(f"Could not read SAR metadata: {e}. Using defaults.")

    # Load scenario config for drift/vessel generation
    try:
        config = _load_scenario_config()
    except FileNotFoundError:
        config = None

    # Build spill result
    spill = {
        "id": spill_id,
        "spill_id": spill_id,
        "name": f"SAR Upload — {file.filename}",
        "centroid_lon": centroid_lon,
        "centroid_lat": centroid_lat,
        "area_km2": area_km2,
        "acquisition_time": datetime.now(timezone.utc).isoformat(),
        "status": "analysed",
        "detection_time": datetime.now(timezone.utc).isoformat(),
        "spill_polygon": _ellipse_geojson(centroid_lon, centroid_lat, 4.0, 1.5, 30),
        "age_estimate": {
            "age_hours_min": 6.0,
            "age_hours_max": 18.0,
            "age_hours_mean": 12.0,
            "method": "fay_spreading",
        },
    }

    # Build drift
    drift_cfg = {
        "wind_speed_ms": 5.2, "wind_direction_degrees": 225,
        "current_speed_ms": 0.3, "current_direction_degrees": 180,
        "hindcast_hours": 48, "forecast_hours": 72,
    }
    full_config = {
        "scenario": {"name": f"Upload: {file.filename}"},
        "spill": {**spill, "major_axis_km": 4.0, "minor_axis_km": 1.5, "orientation_degrees": 30},
        "drift": drift_cfg,
        "ais": config["ais"] if config else {"vessels": []},
    }
    drift = _build_mock_drift(full_config, spill_id)
    vessels = _build_mock_vessels(full_config, drift["origin_lon"], drift["origin_lat"]) if config else []

    return {
        "message": f"Detection complete for {file.filename}",
        "spill_id": spill_id,
        "spill": spill,
        "drift": drift,
        "vessels": vessels,
        "scenario_name": f"Upload: {file.filename}",
    }

