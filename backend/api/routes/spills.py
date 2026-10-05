from fastapi import APIRouter, HTTPException
from typing import List
from backend.api.schemas import SpillSummary, SpillDetail
from backend.api.data_loader import get_spill_summary, load_spill_geojson

router = APIRouter()

@router.get("", response_model=List[SpillSummary])
def list_spills():
    return [get_spill_summary()]

@router.get("/{spill_id}", response_model=SpillDetail)
def get_spill_detail(spill_id: str):
    if spill_id != "spill-001":
        raise HTTPException(status_code=404, detail="Spill not found")
        
    summary = get_spill_summary()
    data = load_spill_geojson()
    props = data["properties"]
    
    return {
        **summary,
        "detection_polygon": data.get("detection_polygon") or {},
        "origin_polygon": data.get("origin_polygon") or {},
        "origin_point": data.get("origin_point") or {},
        "uncertainty_radius_km": props.get("uncertainty_radius_km", 0.0),
        "source": props.get("source", "Unknown"),
        "polarization": props.get("polarization", "Unknown"),
        "thickness_class": props.get("estimated_thickness_class", "Unknown")
    }
