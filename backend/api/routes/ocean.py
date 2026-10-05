from fastapi import APIRouter
from typing import List
from datetime import datetime
from backend.api.schemas import OceanGridPoint
from backend.api.data_loader import load_ocean_data

router = APIRouter()

@router.get("/grid", response_model=List[OceanGridPoint])
def get_ocean_grid(t: datetime):
    ocean_df = load_ocean_data()
    if ocean_df is None or ocean_df.empty:
        return []
        
    ocean_df["time_diff"] = abs((ocean_df["datetime_utc"].dt.tz_localize(None) - t.replace(tzinfo=None)).dt.total_seconds())
    min_diff = ocean_df["time_diff"].min()
    nearest_time = ocean_df[ocean_df["time_diff"] == min_diff].iloc[0]["datetime_utc"]
    
    filtered_df = ocean_df[ocean_df["datetime_utc"] == nearest_time]
    
    results = []
    for _, row in filtered_df.iterrows():
        results.append({
            "datetime_utc": row["datetime_utc"],
            "lat": float(row["lat"]),
            "lon": float(row["lon"]),
            "current_u": float(row["current_u_mps"]),
            "current_v": float(row["current_v_mps"]),
            "wind_u": float(row["wind_u_10m_mps"]),
            "wind_v": float(row["wind_v_10m_mps"])
        })
        
    return results
