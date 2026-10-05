"""
Drift routes: POST /drift/{spill_id}, GET /drift/{spill_id}
"""
import json
import asyncio
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from loguru import logger
from backend.api.db import get_db
from backend.db.models import SpillRecord, DriftResult
from backend.api.schemas import DriftResponse
from backend.drift.drift_engine import DriftEngine
from backend.drift.hindcast import HindcastRunner
from backend.drift.forecast import ForecastRunner
from backend.drift.origin_estimator import OriginEstimator

router = APIRouter()

def _compute_drift_sync(spill_data: dict) -> dict:
    """Synchronous drift computation to run in executor."""
    engine = DriftEngine(use_synthetic=True)
    # Mock forcing data
    forcing = {
        'u_wind': -2.0,
        'v_wind': -3.0,
        'u_current': -0.1,
        'v_current': -0.2,
    }
    hc_runner = HindcastRunner(engine)
    fc_runner = ForecastRunner(engine)
    
    hc = hc_runner.run(spill_data, forcing, duration_hours=48)
    fc = fc_runner.run(spill_data, forcing, duration_hours=72)
    origin = OriginEstimator().estimate(hc)
    
    return {
        'origin_lon': origin['origin_lon'],
        'origin_lat': origin['origin_lat'],
        'origin_time': origin['origin_time'],
        'uncertainty_km': origin['uncertainty_km'],
        'confidence': origin['confidence'],
        'hindcast_geojson': hc['trajectories'],
        'forecast_geojson': fc['trajectories'],
    }

@router.post('/drift/{spill_id}', response_model=DriftResponse)
async def run_drift(spill_id: str, db: AsyncSession = Depends(get_db)):
    record = await db.get(SpillRecord, spill_id)
    if not record:
        raise HTTPException(404, f'Spill {spill_id} not found')
        
    # Delete existing drift result if present
    result = await db.execute(select(DriftResult).where(DriftResult.spill_id == spill_id))
    existing_drift = result.scalars().first()
    if existing_drift:
        await db.delete(existing_drift)
        await db.flush()

    spill_data = {
        'spill_id': record.id,
        'centroid_lon': record.centroid_lon,
        'centroid_lat': record.centroid_lat,
        'area_km2': record.area_km2,
        'acquisition_time': record.acquisition_time,
        'spill_polygon': json.loads(record.spill_polygon_json) if record.spill_polygon_json else None
    }

    loop = asyncio.get_event_loop()
    drift_output = await loop.run_in_executor(None, _compute_drift_sync, spill_data)

    drift_record = DriftResult(
        spill_id=spill_id,
        hindcast_geojson=json.dumps(drift_output['hindcast_geojson']),
        forecast_geojson=json.dumps(drift_output['forecast_geojson']),
        origin_lon=drift_output['origin_lon'],
        origin_lat=drift_output['origin_lat'],
        origin_time=drift_output['origin_time'],
        uncertainty_km=drift_output['uncertainty_km'],
        confidence=drift_output['confidence']
    )
    db.add(drift_record)
    await db.flush()

    return DriftResponse(
        spill_id=spill_id,
        origin_lon=drift_output['origin_lon'],
        origin_lat=drift_output['origin_lat'],
        origin_time=drift_output['origin_time'],
        uncertainty_km=drift_output['uncertainty_km'],
        confidence=drift_output['confidence'],
        hindcast_geojson=drift_output['hindcast_geojson'],
        forecast_geojson=drift_output['forecast_geojson']
    )

@router.get('/drift/{spill_id}', response_model=DriftResponse)
async def get_drift(spill_id: str, db: AsyncSession = Depends(get_db)):
    result = await db.execute(select(DriftResult).where(DriftResult.spill_id == spill_id))
    record = result.scalars().first()
    if not record:
        raise HTTPException(404, f'Drift result for spill {spill_id} not found')
        
    return DriftResponse(
        spill_id=spill_id,
        origin_lon=record.origin_lon,
        origin_lat=record.origin_lat,
        origin_time=record.origin_time,
        uncertainty_km=record.uncertainty_km,
        confidence=record.confidence,
        hindcast_geojson=json.loads(record.hindcast_geojson) if record.hindcast_geojson else {},
        forecast_geojson=json.loads(record.forecast_geojson) if record.forecast_geojson else {}
    )
