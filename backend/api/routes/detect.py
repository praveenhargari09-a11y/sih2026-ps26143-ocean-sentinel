"""
Detection routes: POST /detect, GET /spills, GET /spills/{spill_id}
"""
import json
from datetime import datetime, timezone
from pathlib import Path
from fastapi import APIRouter, UploadFile, File, Form, Depends, HTTPException, Request
from fastapi import Body
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from loguru import logger
from backend.api.db import get_db
from backend.db.models import SpillRecord
from backend.api.schemas import SpillDetectionResponse, SpillSummary
from backend.detection.inference import SpillDetector
from backend.detection.age_estimator import SpillAgeEstimator

router = APIRouter()
PROJECT_ROOT = Path(__file__).resolve().parents[3]
RAW_DIR = PROJECT_ROOT / 'data' / 'raw'
SYNTH_CONFIG = PROJECT_ROOT / 'data' / 'synthetic' / 'scenario_config.yaml'

detector = SpillDetector(model_path=None)  # mock mode — swap in real checkpoint when available
estimator = SpillAgeEstimator()

@router.post('/detect', response_model=SpillDetectionResponse, summary='Detect oil spill from SAR scene')
async def detect_spill(
    request: Request,
    file: UploadFile = File(None),
    name: str = Form(default='Unnamed Spill'),
    sar_path: str = Form(default=None),
    synthetic: bool = Form(default=True),
    db: AsyncSession = Depends(get_db),
):
    """
    Upload a SAR GeoTIFF or provide a server-side path.
    If synthetic=True or no file provided, generates a synthetic scene from scenario_config.yaml.
    """
    import uuid
    spill_id = str(uuid.uuid4())[:8]
    
    # Save uploaded file if provided
    if file and file.filename:
        save_path = RAW_DIR / f'{spill_id}_{file.filename}'
        save_path.parent.mkdir(parents=True, exist_ok=True)
        content = await file.read()
        with open(save_path, 'wb') as f:
            f.write(content)
        actual_path = str(save_path)
        logger.info(f'Saved uploaded SAR file: {save_path}')
    elif sar_path:
        actual_path = sar_path
    else:
        actual_path = None  # will use synthetic mode
    
    # Run detection
    try:
        if actual_path:
            result = detector.detect(actual_path)
        else:
            result = detector.detect_synthetic({'spill_id': spill_id})
    except Exception as e:
        logger.error(f'Detection failed: {e}')
        raise HTTPException(500, f'Detection failed: {str(e)}')
    
    props = result.get('geometry_props', [{}])
    area = props[0].get('area_km2', 10.0) if props else 10.0
    perimeter = props[0].get('perimeter_km', 15.0) if props else 15.0
    centroid_lon = props[0].get('centroid_lon', 68.5) if props else 68.5
    centroid_lat = props[0].get('centroid_lat', 20.3) if props else 20.3
    spill_polygon = result.get('spill_polygons', [None])[0]
    
    age = estimator.estimate_age(area)
    acq_time = result.get('metadata', {}).get('acquisition_time', datetime.now(timezone.utc).isoformat())
    now = datetime.now(timezone.utc).isoformat()
    
    # Save to DB
    record = SpillRecord(
        id=spill_id, name=name, sar_path=actual_path,
        acquisition_time=str(acq_time), detection_time=now,
        centroid_lon=centroid_lon, centroid_lat=centroid_lat,
        area_km2=area, perimeter_km=perimeter,
        age_hours_mean=age['age_hours_mean'],
        spill_polygon_json=json.dumps(spill_polygon) if spill_polygon else None,
        status='analysed'
    )
    db.add(record)
    await db.flush()
    
    return SpillDetectionResponse(
        spill_id=spill_id, name=name,
        centroid_lon=centroid_lon, centroid_lat=centroid_lat,
        area_km2=area, perimeter_km=perimeter,
        age_estimate=age, spill_polygon=spill_polygon,
        status='analysed', acquisition_time=str(acq_time), detection_time=now
    )

@router.get('/spills', response_model=list[SpillSummary], summary='List all analysed spills')
async def list_spills(db: AsyncSession = Depends(get_db)):
    result = await db.execute(select(SpillRecord).order_by(SpillRecord.created_at.desc()))
    records = result.scalars().all()
    return [
        SpillSummary(
            spill_id=r.id, name=r.name, centroid_lon=r.centroid_lon,
            centroid_lat=r.centroid_lat, area_km2=r.area_km2,
            status=r.status, acquisition_time=r.acquisition_time,
            created_at=r.created_at.isoformat() if r.created_at else None
        ) for r in records
    ]

@router.get('/spills/{spill_id}', response_model=SpillDetectionResponse)
async def get_spill(spill_id: str, db: AsyncSession = Depends(get_db)):
    record = await db.get(SpillRecord, spill_id)
    if not record:
        raise HTTPException(404, f'Spill {spill_id} not found')
    return SpillDetectionResponse(
        spill_id=record.id, name=record.name,
        centroid_lon=record.centroid_lon or 0, centroid_lat=record.centroid_lat or 0,
        area_km2=record.area_km2 or 0,
        spill_polygon=json.loads(record.spill_polygon_json) if record.spill_polygon_json else None,
        status=record.status, acquisition_time=record.acquisition_time
    )
