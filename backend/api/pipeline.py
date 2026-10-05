"""
OilSpillPipeline — orchestrates detection → drift → AIS modules.
Progress is reported via an async callback for WebSocket streaming.
"""
import os, json
from datetime import datetime, timezone
from typing import Callable, Awaitable, Optional
from loguru import logger
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SYNTH_CONFIG = PROJECT_ROOT / 'data' / 'synthetic' / 'scenario_config.yaml'

class OilSpillPipeline:
    async def run_full_pipeline(
        self,
        spill_id: str,
        sar_path: Optional[str],
        config: dict,
        progress_cb: Callable[[str, int, str], Awaitable[None]]
    ) -> dict:
        """
        Run the full 3-stage pipeline.
        Stage 1 (0-30%): SAR detection
        Stage 2 (30-65%): Drift hindcast + forecast + origin estimation
        Stage 3 (65-100%): AIS loading, filtering, vessel ranking
        """
        await progress_cb('detection', 5, 'Starting SAR detection...')
        spill = await self._run_detection(sar_path, config)
        await progress_cb('detection', 30, f'Detected spill: {spill["area_km2"]:.1f} km²')
        
        await progress_cb('drift', 35, 'Fetching oceanographic data...')
        drift = await self._run_drift(spill, config)
        await progress_cb('drift', 65, f'Origin estimated at ({drift["origin_lon"]:.3f}, {drift["origin_lat"]:.3f})')
        
        await progress_cb('vessels', 70, 'Loading AIS data and scoring vessels...')
        vessels = await self._run_ais(spill, drift, config)
        await progress_cb('vessels', 100, f'Ranked {len(vessels)} suspect vessels')
        
        return {'spill': spill, 'drift': drift, 'vessels': vessels}
    
    async def _run_detection(self, sar_path, config) -> dict:
        """Run SpillDetector in async thread executor."""
        import asyncio
        from backend.detection.inference import SpillDetector
        from backend.detection.age_estimator import SpillAgeEstimator
        
        loop = asyncio.get_event_loop()
        
        def _detect():
            detector = SpillDetector(model_path=None)  # mock mode
            if sar_path and os.path.exists(sar_path):
                result = detector.detect(sar_path)
            else:
                # Generate synthetic and detect
                result = detector.detect_synthetic(config)
            return result
        
        result = await loop.run_in_executor(None, _detect)
        
        estimator = SpillAgeEstimator()
        polygons = result.get('spill_polygons', [])
        props = result.get('geometry_props', [{}])
        area = props[0].get('area_km2', 10.0) if props else 10.0
        centroid_lon = props[0].get('centroid_lon', config.get('centroid_lon', 68.5)) if props else 68.5
        centroid_lat = props[0].get('centroid_lat', config.get('centroid_lat', 20.3)) if props else 20.3
        age = estimator.estimate_age(area)
        
        return {
            'spill_id': config.get('spill_id', 'demo'),
            'centroid_lon': centroid_lon,
            'centroid_lat': centroid_lat,
            'area_km2': area,
            'perimeter_km': props[0].get('perimeter_km', 0) if props else 0,
            'spill_polygon': polygons[0] if polygons else None,
            'age_estimate': age,
            'acquisition_time': config.get('acquisition_time', datetime.now(timezone.utc).isoformat()),
        }
    
    async def _run_drift(self, spill, config) -> dict:
        import asyncio
        from backend.drift.drift_engine import DriftEngine
        from backend.drift.hindcast import HindcastRunner
        from backend.drift.forecast import ForecastRunner
        from backend.drift.origin_estimator import OriginEstimator
        
        loop = asyncio.get_event_loop()
        
        def _drift():
            engine = DriftEngine(use_synthetic=True)
            forcing = {
                'u_wind': config.get('u_wind', -2.0),
                'v_wind': config.get('v_wind', -3.0),
                'u_current': config.get('u_current', -0.1),
                'v_current': config.get('v_current', -0.2),
            }
            hc_runner = HindcastRunner(engine)
            fc_runner = ForecastRunner(engine)
            
            hc = hc_runner.run(spill, forcing, duration_hours=48)
            fc = fc_runner.run(spill, forcing, duration_hours=72)
            origin = OriginEstimator().estimate(hc)
            
            return {
                'spill_id': spill.get('spill_id', 'demo'),
                'origin_lon': origin['origin_lon'],
                'origin_lat': origin['origin_lat'],
                'origin_time': origin['origin_time'],
                'uncertainty_km': origin['uncertainty_km'],
                'confidence': origin['confidence'],
                'hindcast_geojson': hc['trajectories'],
                'forecast_geojson': fc['trajectories'],
                'extent_polygons': fc.get('extent_polygons', []),
            }
        
        return await loop.run_in_executor(None, _drift)
    
    async def _run_ais(self, spill, drift, config) -> list:
        import asyncio
        loop = asyncio.get_event_loop()
        
        def _ais():
            from backend.ais.ais_synth import generate_demo_ais
            from backend.ais.spatial_filter import SpatialFilter
            from backend.ais.vessel_ranker import VesselRanker
            
            ais_gdf = generate_demo_ais(str(SYNTH_CONFIG))
            candidates = SpatialFilter().filter_by_origin(
                ais_gdf,
                origin_lon=drift['origin_lon'],
                origin_lat=drift['origin_lat'],
                origin_time_utc=drift['origin_time'],
                search_radius_km=200,
                time_window_hours=72
            )
            return VesselRanker().rank(candidates, spill, drift)
        
        return await loop.run_in_executor(None, _ais)
