from fastapi import APIRouter
from typing import List
from backend.api.schemas import VesselScore
from backend.api.data_loader import compute_vessel_scores

router = APIRouter()

@router.get("/{spill_id}", response_model=List[VesselScore])
def list_vessel_scores(spill_id: str):
    return compute_vessel_scores(spill_id)
