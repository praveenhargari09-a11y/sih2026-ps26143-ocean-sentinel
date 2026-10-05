"""
Hindcast runner — seeds particles at the detected spill location and runs
the drift engine backward in time to identify the oil spill origin.
"""
import math
from datetime import datetime, timedelta, timezone
from typing import Optional
from loguru import logger
from .drift_engine import DriftEngine

class HindcastRunner:
    def __init__(self, drift_engine: Optional[DriftEngine] = None):
        self.engine = drift_engine or DriftEngine(use_synthetic=True)

    def run(self, spill_info: dict, forcing: dict, duration_hours: int = 48) -> dict:
        """
        Run backward drift simulation from spill centroid.
        
        Args:
            spill_info: dict with centroid_lon, centroid_lat, acquisition_time (ISO str)
            forcing: dict with u_wind, v_wind, u_current, v_current (m/s scalars)
            duration_hours: how many hours to trace backward
        Returns:
            dict with 'trajectories' (GeoJSON FeatureCollection) and 'origin_candidates' (list of dicts)
        """
        lon = spill_info['centroid_lon']
        lat = spill_info['centroid_lat']
        acq_time = spill_info.get('acquisition_time', datetime.now(timezone.utc).isoformat())
        if isinstance(acq_time, str):
            acq_time = datetime.fromisoformat(acq_time.replace('Z', '+00:00'))
        
        logger.info(f"Running hindcast from ({lon:.4f}, {lat:.4f}) for {duration_hours}h backward")
        trajectories = self.engine.run_hindcast(
            lon=lon, lat=lat,
            time_satellite=acq_time,
            duration_hours=duration_hours,
            n_particles=200
        )
        
        # Extract endpoint positions as origin candidates
        origin_candidates = self._extract_endpoints(trajectories)
        
        return {
            'trajectories': trajectories,
            'origin_candidates': origin_candidates
        }
    
    def _extract_endpoints(self, trajectories: dict) -> list:
        """Extract the final (earliest in time = backward endpoint) positions of all particle tracks."""
        endpoints = []
        for feature in trajectories.get('features', []):
            geom = feature.get('geometry', {})
            if geom.get('type') == 'LineString':
                coords = geom.get('coordinates', [])
                if coords:
                    lon, lat = coords[-1][0], coords[-1][1]
                    endpoints.append({'lon': lon, 'lat': lat})
        return endpoints
