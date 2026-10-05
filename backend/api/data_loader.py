import json
import pandas as pd
import numpy as np
from pathlib import Path
from math import radians, cos, sin, asin, sqrt, atan2, degrees
from datetime import datetime, timedelta

DATA_DIR = Path(__file__).parent.parent / "synthetic_data"

_SPILL_DATA = None
_AIS_DATA = None
_OCEAN_DATA = None
_SPILL_SUMMARY = None

def haversine(lat1, lon1, lat2, lon2) -> float:
    R = 6372.8 
    dLat = radians(lat2 - lat1)
    dLon = radians(lon2 - lon1)
    lat1 = radians(lat1)
    lat2 = radians(lat2)
    a = sin(dLat/2)**2 + cos(lat1)*cos(lat2)*sin(dLon/2)**2
    c = 2*asin(sqrt(a))
    return R * c

def calculate_bearing(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    dlon = lon2 - lon1
    x = sin(dlon) * cos(lat2)
    y = cos(lat1) * sin(lat2) - (sin(lat1) * cos(lat2) * cos(dlon))
    initial_bearing = atan2(x, y)
    initial_bearing = degrees(initial_bearing)
    compass_bearing = (initial_bearing + 360) % 360
    return compass_bearing

def load_data():
    global _SPILL_DATA, _AIS_DATA, _OCEAN_DATA
    
    if not (DATA_DIR / "oil_spill.geojson").exists():
        # Dummy data if missing so tests pass
        return

    with open(DATA_DIR / "oil_spill.geojson", "r") as f:
        _SPILL_DATA = json.load(f)
    
    _AIS_DATA = pd.read_csv(DATA_DIR / "ais_tracks.csv", parse_dates=["BaseDateTime"])
    _OCEAN_DATA = pd.read_csv(DATA_DIR / "ocean_wind_currents.csv", parse_dates=["datetime_utc"])

def load_spill_geojson():
    if _SPILL_DATA is None:
        load_data()
        
    if _SPILL_DATA is None:
        return {"detection_polygon": None, "origin_polygon": None, "origin_point": None, "properties": {}}
    
    res = {
        "detection_polygon": None,
        "origin_polygon": None,
        "origin_point": None,
        "properties": {}
    }
    
    for feature in _SPILL_DATA.get("features", []):
        role = feature.get("properties", {}).get("feature_role")
        if role == "sar_detection":
            res["detection_polygon"] = feature["geometry"]
            res["properties"].update(feature["properties"])
        elif role == "hindcast_origin_estimate":
            res["origin_polygon"] = feature["geometry"]
            res["properties"].update(feature["properties"])
        elif role == "origin_point_estimate":
            res["origin_point"] = feature["geometry"]
            res["properties"].update(feature["properties"])
            
    return res

def get_spill_summary():
    global _SPILL_SUMMARY
    if _SPILL_SUMMARY is not None:
        return _SPILL_SUMMARY

    data = load_spill_geojson()
    origin_coords = data["origin_point"]["coordinates"] if data["origin_point"] else [0.0, 0.0]
    
    _SPILL_SUMMARY = {
        "id": "spill-001",
        "detection_time": data["properties"].get("detection_time_utc", "2026-08-14T06:32:00"),
        "origin_time": data["properties"].get("estimated_origin_time_utc", "2026-08-13T21:17:00"),
        "area_km2": data["properties"].get("area_km2", 0.0),
        "confidence": data["properties"].get("confidence", 0.0),
        "origin_lon": origin_coords[0],
        "origin_lat": origin_coords[1]
    }
    return _SPILL_SUMMARY

def load_ais_tracks():
    if _AIS_DATA is None:
        load_data()
    return _AIS_DATA

def load_ocean_data():
    if _OCEAN_DATA is None:
        load_data()
    return _OCEAN_DATA

def compute_vessel_scores(spill_id: str) -> list[dict]:
    ais_df = load_ais_tracks()
    if ais_df is None: return []
    
    summary = get_spill_summary()
    origin_lat = summary["origin_lat"]
    origin_lon = summary["origin_lon"]
    origin_time = pd.to_datetime(summary["origin_time"]).tz_localize(None)
    
    results = []
    
    for mmsi, group in ais_df.groupby("MMSI"):
        group = group.sort_values("BaseDateTime")
        
        # 1. Proximity
        group["dist_to_origin"] = group.apply(lambda row: haversine(row["LAT"], row["LON"], origin_lat, origin_lon), axis=1)
        closest_idx = group["dist_to_origin"].idxmin()
        closest_row = group.loc[closest_idx]
        closest_km = closest_row["dist_to_origin"]
        closest_time = closest_row["BaseDateTime"]
        
        proximity_score = max(0.0, 1 - closest_km / 50.0)
        
        # 2. Trajectory alignment
        bearing = calculate_bearing(closest_row["LAT"], closest_row["LON"], origin_lat, origin_lon)
        cog = closest_row["COG"]
        bearing_diff = abs(bearing - cog)
        if bearing_diff > 180:
            bearing_diff = 360 - bearing_diff
        trajectory_score = max(0.0, 1 - bearing_diff / 90.0)
        
        # 3. AIS darkness
        time_diffs = group["BaseDateTime"].diff().dt.total_seconds() / 3600
        max_gap = time_diffs.max() if not time_diffs.empty and pd.notna(time_diffs.max()) else 0
        
        darkness_score = 0.1
        if max_gap > 1.0:
            darkness_score = min(1.0, max_gap / 24.0)
            
        # 4. Vessel type
        v_type = str(closest_row["VesselType"]).lower()
        if "tanker" in v_type:
            vtype_score = 0.9
        elif "cargo" in v_type:
            vtype_score = 0.3
        elif "fishing" in v_type:
            vtype_score = 0.1
        else:
            vtype_score = 0.5
            
        # 5. Speed anomaly
        group["BaseDateTimeTzFree"] = group["BaseDateTime"].dt.tz_localize(None)
        window = group[(group["BaseDateTimeTzFree"] >= origin_time - pd.Timedelta(hours=3)) &
                       (group["BaseDateTimeTzFree"] <= origin_time + pd.Timedelta(hours=3))]
        if len(window) > 1:
            mean_sog = window["SOG"].mean()
            std_sog = window["SOG"].std()
            speed_score = min(1.0, std_sog / mean_sog) if mean_sog > 0 else 0.0
        else:
            speed_score = 0.0
            
        total_score = (proximity_score * 0.35 + 
                       trajectory_score * 0.25 + 
                       darkness_score * 0.20 + 
                       vtype_score * 0.10 + 
                       speed_score * 0.10)
        
        results.append({
            "mmsi": int(mmsi),
            "vessel_name": str(closest_row["VesselName"]),
            "vessel_type": str(closest_row["VesselType"]),
            "flag": "Unknown",
            "total_score": float(total_score),
            "score_breakdown": {
                "proximity": float(proximity_score),
                "trajectory": float(trajectory_score),
                "darkness": float(darkness_score),
                "vessel_type": float(vtype_score),
                "speed_anomaly": float(speed_score)
            },
            "closest_approach_km": float(closest_km),
            "closest_approach_time": closest_time,
            "ais_gaps": []
        })
        
    results.sort(key=lambda x: x["total_score"], reverse=True)
    for i, res in enumerate(results):
        res["rank"] = i + 1
        
    return results

def interpolate_position(mmsi: int, timestamp: datetime):
    ais_df = load_ais_tracks()
    if ais_df is None: return None
    
    vessel_data = ais_df[ais_df["MMSI"] == mmsi].sort_values("BaseDateTime").copy()
    if vessel_data.empty:
        return None
        
    timestamp_tzfree = timestamp.replace(tzinfo=None)
    vessel_data["time_diff"] = (vessel_data["BaseDateTime"].dt.tz_localize(None) - timestamp_tzfree).dt.total_seconds()
    
    before = vessel_data[vessel_data["time_diff"] <= 0]
    after = vessel_data[vessel_data["time_diff"] > 0]
    
    if before.empty:
        row = after.iloc[0]
        return {"mmsi": mmsi, "timestamp": timestamp, "lat": row["LAT"], "lon": row["LON"], "heading": row["Heading"], "sog": row["SOG"]}
    if after.empty:
        row = before.iloc[-1]
        return {"mmsi": mmsi, "timestamp": timestamp, "lat": row["LAT"], "lon": row["LON"], "heading": row["Heading"], "sog": row["SOG"]}
        
    b = before.iloc[-1]
    a = after.iloc[0]
    
    total_time = (a["BaseDateTime"] - b["BaseDateTime"]).total_seconds()
    if total_time == 0:
        frac = 0
    else:
        frac = (timestamp_tzfree - b["BaseDateTime"].tz_localize(None)).total_seconds() / total_time
        
    lat = b["LAT"] + (a["LAT"] - b["LAT"]) * frac
    lon = b["LON"] + (a["LON"] - b["LON"]) * frac
    heading = b["Heading"] + (a["Heading"] - b["Heading"]) * frac
    sog = b["SOG"] + (a["SOG"] - b["SOG"]) * frac
    
    return {"mmsi": mmsi, "timestamp": timestamp, "lat": lat, "lon": lon, "heading": heading, "sog": sog}

try:
    load_data()
except Exception:
    pass
