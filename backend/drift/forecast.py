"""
Forecast runner — projects the spill forward in time to predict future extent.
"""
import math
from datetime import datetime, timedelta, timezone
from typing import Optional, List
from loguru import logger
from .drift_engine import DriftEngine
import shapely.geometry
from shapely.geometry import MultiPoint

class ForecastRunner:
    def __init__(self, drift_engine: Optional[DriftEngine] = None):
        self.engine = drift_engine or DriftEngine(use_synthetic=True)

    def run(self, spill_info: dict, forcing: dict, duration_hours: int = 48) -> dict:
        """
        Run forward drift simulation from spill centroid.
        
        Args:
            spill_info: dict with centroid_lon, centroid_lat, acquisition_time (ISO str)
            forcing: dict with u_wind, v_wind, u_current, v_current (m/s scalars)
            duration_hours: how many hours to project forward
        Returns:
            dict with 'trajectories' (GeoJSON FeatureCollection) and 'extent_polygons' (list of GeoJSON Features)
        """
        lon = spill_info['centroid_lon']
        lat = spill_info['centroid_lat']
        acq_time = spill_info.get('acquisition_time', datetime.now(timezone.utc).isoformat())
        if isinstance(acq_time, str):
            acq_time = datetime.fromisoformat(acq_time.replace('Z', '+00:00'))
        
        logger.info(f"Running forecast from ({lon:.4f}, {lat:.4f}) for {duration_hours}h forward")
        trajectories = self.engine.run_forecast(
            lon=lon, lat=lat,
            time_satellite=acq_time,
            duration_hours=duration_hours,
            n_particles=200
        )
        
        extent_polygons = self._build_extent_polygons(trajectories, n_steps=6)
        
        return {
            'trajectories': trajectories,
            'extent_polygons': extent_polygons
        }
    
    def _build_extent_polygons(self, trajectories: dict, n_steps: int = 6) -> List[dict]:
        """Divide the time into n_steps and build a convex hull polygon around particle positions at each step."""
        features = trajectories.get('features', [])
        if not features:
            return []
            
        # Collect coords per step
        steps_coords = [[] for _ in range(n_steps)]
        
        for feature in features:
            geom = feature.get('geometry', {})
            if geom.get('type') == 'LineString':
                coords = geom.get('coordinates', [])
                # Assume coords are uniformly distributed over duration
                if coords:
                    n_coords = len(coords)
                    for i in range(n_steps):
                        idx = int(i * (n_coords - 1) / max(1, n_steps - 1))
                        if idx < n_coords:
                            steps_coords[i].append((coords[idx][0], coords[idx][1]))
                            
        extent_features = []
        for i, pts in enumerate(steps_coords):
            if len(pts) >= 3:
                multipoint = MultiPoint(pts)
                hull = multipoint.convex_hull
                
                if hull.geom_type == 'Polygon':
                    extent_features.append({
                        'type': 'Feature',
                        'properties': {'step': i},
                        'geometry': shapely.geometry.mapping(hull)
                    })
        return extent_features
