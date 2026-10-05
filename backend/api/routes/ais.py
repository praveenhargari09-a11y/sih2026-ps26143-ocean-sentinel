from fastapi import APIRouter, HTTPException
from datetime import datetime
from backend.api.schemas import AISTrack, InterpolatedPosition
from backend.api.data_loader import load_ais_tracks, interpolate_position as interpolate_pos

router = APIRouter()

@router.get("/{mmsi}", response_model=AISTrack)
def get_ais_track(mmsi: int):
    ais_df = load_ais_tracks()
    if ais_df is None: raise HTTPException(status_code=404, detail="No data")
    
    vessel_data = ais_df[ais_df["MMSI"] == mmsi].sort_values("BaseDateTime")
    if vessel_data.empty:
        raise HTTPException(status_code=404, detail="Vessel not found")
        
    positions = []
    coords = []
    for _, row in vessel_data.iterrows():
        positions.append({
            "mmsi": int(row["MMSI"]),
            "vessel_name": str(row["VesselName"]),
            "vessel_type": str(row["VesselType"]),
            "timestamp": row["BaseDateTime"],
            "lat": float(row["LAT"]),
            "lon": float(row["LON"]),
            "sog": float(row["SOG"]),
            "cog": float(row["COG"]),
            "heading": float(row["Heading"]),
            "status": str(row["Status"])
        })
        coords.append([float(row["LON"]), float(row["LAT"])])
        
    return {
        "mmsi": mmsi,
        "vessel_name": positions[0]["vessel_name"],
        "vessel_type": positions[0]["vessel_type"],
        "positions": positions,
        "track_geojson": {
            "type": "LineString",
            "coordinates": coords
        }
    }

@router.get("/{mmsi}/at", response_model=InterpolatedPosition)
def get_interpolated_position(mmsi: int, t: datetime):
    pos = interpolate_pos(mmsi, t)
    if not pos:
        raise HTTPException(status_code=404, detail="Vessel not found")
    return pos
