from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from datetime import datetime

class SpillSummary(BaseModel):
    id: str
    detection_time: datetime
    origin_time: datetime
    area_km2: float
    confidence: float
    origin_lat: float
    origin_lon: float

class SpillDetail(SpillSummary):
    detection_polygon: dict
    origin_polygon: dict
    origin_point: dict
    uncertainty_radius_km: float
    source: str
    polarization: str
    thickness_class: str

class VesselScore(BaseModel):
    rank: int
    mmsi: int
    vessel_name: str
    vessel_type: str
    flag: str
    total_score: float
    score_breakdown: Dict[str, float]
    closest_approach_km: float
    closest_approach_time: datetime
    ais_gaps: List[str]

class AISPosition(BaseModel):
    mmsi: int
    vessel_name: str
    vessel_type: str
    timestamp: datetime
    lat: float
    lon: float
    sog: float
    cog: float
    heading: float
    status: str

class AISTrack(BaseModel):
    mmsi: int
    vessel_name: str
    vessel_type: str
    positions: List[AISPosition]
    track_geojson: dict

class InterpolatedPosition(BaseModel):
    mmsi: int
    timestamp: datetime
    lat: float
    lon: float
    heading: float
    sog: float

class OceanGridPoint(BaseModel):
    datetime_utc: datetime
    lat: float
    lon: float
    current_u: float
    current_v: float
    wind_u: float
    wind_v: float

class PipelineEvent(BaseModel):
    spill_id: str
    stage: str
    status: str
    progress_pct: int
    message: str
    timestamp: datetime
