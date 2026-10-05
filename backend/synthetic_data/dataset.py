"""
Synthetic dataset generator for Oil Spill Drift Prediction & Vessel Attribution
Region: Arabian Sea, off Mumbai High offshore oil field, Indian west coast
Scenario: A SAR pass detects an oil slick. Multiple vessels were in the area in the
preceding 48h. One tanker ("suspect") shows a track that passes through the spill
origin near the estimated spill time, followed by an anomalous course change
(possible illicit discharge behaviour). Other vessels are decoys (fishing boats,
cargo ships on unrelated routes, a legitimate tanker on a scheduled route).
"""

import csv
import json
import random
import math
from datetime import datetime, timedelta

random.seed(42)

# ---------------------------------------------------------------------------
# Scenario anchor points
# ---------------------------------------------------------------------------
SPILL_ORIGIN = (19.42, 71.58)          # lat, lon -- near Mumbai High offshore field
SPILL_DETECTION_TIME = datetime(2026, 8, 14, 6, 32, 0)   # Sentinel-1 pass, UTC
SPILL_ORIGIN_TIME = SPILL_DETECTION_TIME - timedelta(hours=9, minutes=15)  # hindcast estimate

KNOTS_TO_MPS = 0.514444
EARTH_R_KM = 6371.0


def move_latlon(lat, lon, bearing_deg, distance_km):
    """Move a point given bearing (deg, 0=N) and distance (km)."""
    br = math.radians(bearing_deg)
    lat1 = math.radians(lat)
    lon1 = math.radians(lon)
    d_r = distance_km / EARTH_R_KM
    lat2 = math.asin(math.sin(lat1) * math.cos(d_r) + math.cos(lat1) * math.sin(d_r) * math.cos(br))
    lon2 = lon1 + math.atan2(
        math.sin(br) * math.sin(d_r) * math.cos(lat1),
        math.cos(d_r) - math.sin(lat1) * math.sin(lat2),
    )
    return math.degrees(lat2), math.degrees(lon2)


# ---------------------------------------------------------------------------
# 1. AIS VESSEL TRACKS  (marinecadastre.gov-style schema)
# ---------------------------------------------------------------------------
vessels = [
    # mmsi, name, type, category, start_offset_hr(before detection), track kind
    (419012345, "MT SAMUDRA VEER", "Tanker", "suspect"),
    (419087654, "MT KOKAN PRIDE",  "Tanker", "legit_scheduled"),
    (563099887, "MV OCEAN GARNET", "Cargo",  "unrelated"),
    (412233445, "FV MATSYA-7",     "Fishing", "unrelated"),
    (412233446, "FV MATSYA-9",     "Fishing", "unrelated"),
    (636099211, "MV PACIFIC LOTUS","Cargo",  "unrelated"),
    (419055678, "MT DEV PRAYAG",   "Tanker", "unrelated_tanker"),
    (525011234, "MV BINTANG LAUT", "Cargo",  "unrelated"),
]

ais_rows = []
track_start = SPILL_DETECTION_TIME - timedelta(hours=48)
track_end = SPILL_DETECTION_TIME + timedelta(hours=12)
step_min = 20

for mmsi, name, vtype, category in vessels:
    t = track_start
    # All starting positions strictly in Arabian Sea (west of Indian coastline, lon <= 72.0)
    if category == "suspect":
        # Suspect approaches origin from SW/W in Arabian Sea
        start_lat, start_lon = move_latlon(*SPILL_ORIGIN, bearing_deg=random.uniform(200, 260), distance_km=random.uniform(35, 50))
        heading = (math.degrees(math.atan2(
            math.radians(SPILL_ORIGIN[1] - start_lon), math.radians(SPILL_ORIGIN[0] - start_lat))) + 360) % 360
        speed = random.uniform(11, 13)
    else:
        # Decoys in open water off Mumbai High
        start_lat, start_lon = move_latlon(*SPILL_ORIGIN, bearing_deg=random.uniform(160, 340), distance_km=random.uniform(25, 90))
        # Ensure starting point is strictly seaward
        if start_lon > 72.0:
            start_lon = 72.0 - random.uniform(0.1, 0.5)
        heading = random.uniform(160, 340)  # generally seaward / along coast
        speed = random.uniform(4, 15) if vtype != "Fishing" else random.uniform(2, 6)

    lat, lon = start_lat, start_lon
    passed_origin = False

    while t <= track_end:
        # suspect vessel: steer toward spill origin, arrive near origin time, then divert sharply
        if category == "suspect":
            if t < SPILL_ORIGIN_TIME:
                brg = (math.degrees(math.atan2(
                    math.radians(SPILL_ORIGIN[1] - lon), math.radians(SPILL_ORIGIN[0] - lat))) + 360) % 360
                heading = brg
                speed = 12.5
            elif not passed_origin:
                passed_origin = True
                speed = 6.0  # slows near origin window (consistent with illicit discharge / tank cleaning)
            else:
                # anomalous sharp course change + speed increase after the window
                heading = (heading + 137) % 360
                speed = 15.5
        else:
            # gentle random walk with Arabian Sea boundary constraints
            heading = (heading + random.uniform(-6, 6)) % 360
            speed = max(0.5, speed + random.uniform(-0.4, 0.4))

            # Coastline of Maharashtra/Goa is at lon ~72.8E; keep well clear (lon <= 72.2)
            if lon > 72.0:
                heading = random.uniform(210, 310)  # steer W/SW into open sea
            elif lon < 66.5:
                heading = random.uniform(30, 130)   # steer E/NE
            if lat > 20.6:
                heading = random.uniform(150, 210)  # steer South
            elif lat < 16.5:
                heading = random.uniform(330, 30)   # steer North

        dist_km = speed * KNOTS_TO_MPS * (step_min * 60) / 1000.0
        lat, lon = move_latlon(lat, lon, heading, dist_km)

        nav_status = "Under way using engine"
        if vtype == "Fishing" and random.random() < 0.15:
            nav_status = "Engaged in fishing"

        ais_rows.append({
            "MMSI": mmsi,
            "VesselName": name,
            "VesselType": vtype,
            "BaseDateTime": t.strftime("%Y-%m-%dT%H:%M:%S"),
            "LAT": round(lat, 5),
            "LON": round(lon, 5),
            "SOG": round(speed, 1),
            "COG": round(heading, 1),
            "Heading": round(heading, 1),
            "Status": nav_status,
            "Category_GroundTruth": category,   # only present because this is synthetic training/eval data
        })
        t += timedelta(minutes=step_min)

import os
from pathlib import Path
OUTPUT_DIR = Path(__file__).parent
ais_path = OUTPUT_DIR / "ais_tracks.csv"
with open(ais_path, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=list(ais_rows[0].keys()))
    writer.writeheader()
    writer.writerows(ais_rows)

# ---------------------------------------------------------------------------
# 2. OIL SPILL DETECTIONS (GeoJSON) -- SAR detection polygon + backtracked origin
# ---------------------------------------------------------------------------
def spill_polygon(center, radius_km, elongation_bearing, n=24, jitter=0.15):
    pts = []
    for i in range(n):
        ang = 360 * i / n
        r = radius_km * (1 + elongation_bearing * math.cos(math.radians(ang)) * 0.4)
        r *= (1 + random.uniform(-jitter, jitter))
        lat, lon = move_latlon(center[0], center[1], ang, max(r, 0.1))
        pts.append([round(lon, 5), round(lat, 5)])
    pts.append(pts[0])
    return pts

detection_center = move_latlon(*SPILL_ORIGIN, bearing_deg=48, distance_km=6.4)  # drifted since origin
origin_poly = spill_polygon(SPILL_ORIGIN, radius_km=0.4, elongation_bearing=1)
detection_poly = spill_polygon(detection_center, radius_km=2.1, elongation_bearing=1)

geojson = {
    "type": "FeatureCollection",
    "features": [
        {
            "type": "Feature",
            "properties": {
                "feature_role": "sar_detection",
                "detection_time_utc": SPILL_DETECTION_TIME.isoformat(),
                "source": "Sentinel-1 SAR (synthetic)",
                "polarization": "VV",
                "area_km2": 9.8,
                "confidence": 0.87,
                "estimated_thickness_class": "sheen-to-medium",
            },
            "geometry": {"type": "Polygon", "coordinates": [detection_poly]},
        },
        {
            "type": "Feature",
            "properties": {
                "feature_role": "hindcast_origin_estimate",
                "estimated_origin_time_utc": SPILL_ORIGIN_TIME.isoformat(),
                "method": "backward particle drift (synthetic)",
                "uncertainty_radius_km": 1.2,
            },
            "geometry": {"type": "Polygon", "coordinates": [origin_poly]},
        },
        {
            "type": "Feature",
            "properties": {"feature_role": "origin_point_estimate"},
            "geometry": {"type": "Point", "coordinates": [SPILL_ORIGIN[1], SPILL_ORIGIN[0]]},
        },
    ],
}

spill_path = OUTPUT_DIR / "oil_spill.geojson"
with open(spill_path, "w") as f:
    json.dump(geojson, f, indent=2)

# ---------------------------------------------------------------------------
# 3. OCEANOGRAPHIC / MET DATA -- current & wind grid for drift modelling
# ---------------------------------------------------------------------------
ocean_rows = []
lat_min, lat_max = 19.0, 19.9
lon_min, lon_max = 71.0, 72.1
grid_step = 0.1

t = SPILL_ORIGIN_TIME - timedelta(hours=3)
while t <= SPILL_DETECTION_TIME + timedelta(hours=6):
    lat = lat_min
    while lat <= lat_max:
        lon = lon_min
        while lon <= lon_max:
            # background SW monsoon-ish current/wind with mild spatial + temporal variation
            current_u = 0.35 + 0.05 * math.sin(lat * 3) + random.uniform(-0.03, 0.03)
            current_v = 0.18 + 0.04 * math.cos(lon * 3) + random.uniform(-0.03, 0.03)
            wind_u = 4.2 + 0.3 * math.sin(lon * 2) + random.uniform(-0.2, 0.2)
            wind_v = 2.1 + 0.3 * math.cos(lat * 2) + random.uniform(-0.2, 0.2)
            ocean_rows.append({
                "datetime_utc": t.strftime("%Y-%m-%dT%H:%M:%S"),
                "lat": round(lat, 2),
                "lon": round(lon, 2),
                "current_u_mps": round(current_u, 3),
                "current_v_mps": round(current_v, 3),
                "wind_u_10m_mps": round(wind_u, 3),
                "wind_v_10m_mps": round(wind_v, 3),
            })
            lon += grid_step
        lat += grid_step
    t += timedelta(hours=3)

ocean_path = OUTPUT_DIR / "ocean_wind_currents.csv"
with open(ocean_path, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=list(ocean_rows[0].keys()))
    writer.writeheader()
    writer.writerows(ocean_rows)

print(f"AIS rows: {len(ais_rows)}")
print(f"Ocean/wind rows: {len(ocean_rows)}")
print(f"Spill origin: {SPILL_ORIGIN}, detection center: {detection_center}")
print("Files written:", str(ais_path), str(spill_path), str(ocean_path))
