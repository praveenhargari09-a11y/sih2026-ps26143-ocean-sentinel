import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
import sys

# Add backend to path so we can import OceanDataFetcher
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../")))
from backend.drift.data_fetcher import OceanDataFetcher

print("Starting scripts...")

vessels = [
    {"mmsi": "563118200", "name": "MV X-Press Pearl", "type": "Container", "guilty": True},
    {"mmsi": "563112345", "name": "MV Pacific Star", "type": "Container", "guilty": False},
    {"mmsi": "419008765", "name": "MV Mumbai Trader", "type": "Bulk Carrier", "guilty": False},
    {"mmsi": "416002345", "name": "MV Lanka Pride", "type": "Cargo", "guilty": False},
    {"mmsi": "538009876", "name": "MV Ocean Venture", "type": "Tanker", "guilty": False},
    {"mmsi": "477005678", "name": "MV Eastern Dragon", "type": "Container", "guilty": False},
]

start_time = datetime(2021, 5, 19, 0, 0, 0)
end_time = datetime(2021, 5, 22, 0, 0, 0)
time_step = timedelta(minutes=10)

times = []
current = start_time
while current <= end_time:
    times.append(current)
    current += time_step

rows = []

# REAL HISTORICAL SPATIAL ORIGIN
TRUE_LAT = 7.0825
TRUE_LON = 79.7775

for v in vessels:
    if v["guilty"]:
        lat = TRUE_LAT
        lon = TRUE_LON
        for t in times:
            if datetime(2021, 5, 20, 2, 0, 0) <= t < datetime(2021, 5, 22, 2, 0, 0):
                continue
            d_lat = np.random.uniform(-0.0001, 0.0001)
            d_lon = np.random.uniform(-0.0001, 0.0001)
            rows.append({
                "MMSI": v["mmsi"],
                "BaseDateTime": t.strftime("%Y-%m-%dT%H:%M:%SZ"),
                "LAT": round(lat + d_lat, 5),
                "LON": round(lon + d_lon, 5),
                "SOG": round(np.random.uniform(0, 0.5), 1),
                "COG": round(np.random.uniform(0, 360), 1),
                "Heading": round(np.random.uniform(0, 360), 1),
                "VesselName": v["name"],
                "VesselType": v["type"],
                "Status": "at anchor"
            })
    else:
        # Spawn decoy vessels in the Laccadive Sea (West of Sri Lanka) to avoid land
        lat = np.random.uniform(6.5, 7.5)
        lon = np.random.uniform(79.0, 79.7)
        d_lat = np.random.uniform(-0.05, 0.05) / 6
        d_lon = np.random.uniform(-0.05, 0.05) / 6
        for t in times:
            lat += d_lat
            lon += d_lon
            sog = round(np.random.uniform(8, 15), 1)
            cog = round(np.random.uniform(0, 360), 1)
            rows.append({
                "MMSI": v["mmsi"],
                "BaseDateTime": t.strftime("%Y-%m-%dT%H:%M:%SZ"),
                "LAT": round(lat, 5),
                "LON": round(lon, 5),
                "SOG": sog,
                "COG": cog,
                "Heading": cog,
                "VesselName": v["name"],
                "VesselType": v["type"],
                "Status": "under way using engine"
            })

df = pd.DataFrame(rows)
df.to_csv("c:/Users/prave/Desktop/Oil Spill/data/incidents/xpress_pearl_2021/ais_traffic.csv", index=False)
print("Saved ais_traffic.csv")

print("Generating historical ocean_currents.nc...")
# Generate the NetCDF using the real historical values from incident.yaml (wind 6.5@240, current 0.35@170)
fetcher = OceanDataFetcher(cache_dir="c:/Users/prave/Desktop/Oil Spill/data/incidents/xpress_pearl_2021")
nc_path = fetcher.get_synthetic_forcing(
    lon=TRUE_LON,
    lat=TRUE_LAT,
    time_start="2021-05-19T00:00:00Z",
    time_end="2021-05-22T00:00:00Z",
    output_path="c:/Users/prave/Desktop/Oil Spill/data/incidents/xpress_pearl_2021/ocean_currents.nc",
    kind="combined",
    wind_speed_ms=6.5,
    wind_dir_deg=240.0,
    current_speed_ms=0.35,
    current_dir_deg=170.0
)
print(f"Saved real historical {nc_path}")
