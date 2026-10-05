"""
Estimates the spill origin from hindcast particle endpoints using DBSCAN clustering.
"""
import numpy as np
from sklearn.cluster import DBSCAN
from loguru import logger
from datetime import datetime, timedelta, timezone

class OriginEstimator:
    def estimate(self, hindcast_result: dict) -> dict:
        """
        Cluster particle endpoints → pick largest cluster → return centroid as origin.
        Returns:
            {'origin_lon', 'origin_lat', 'origin_time', 'uncertainty_km', 'confidence', 'n_particles'}
        """
        endpoints = hindcast_result.get('origin_candidates', [])
        if not endpoints:
            # Fallback: parse from trajectory endpoints
            for feat in hindcast_result.get('trajectories', {}).get('features', []):
                coords = feat.get('geometry', {}).get('coordinates', [])
                if coords:
                    endpoints.append({'lon': coords[-1][0], 'lat': coords[-1][1]})
        
        if len(endpoints) < 3:
            # Not enough particles, return mean
            if endpoints:
                lons = [e['lon'] for e in endpoints]
                lats = [e['lat'] for e in endpoints]
                return {
                    'origin_lon': float(np.mean(lons)),
                    'origin_lat': float(np.mean(lats)),
                    'origin_time': (datetime.now(timezone.utc) - timedelta(hours=24)).isoformat(),
                    'uncertainty_km': 50.0,
                    'confidence': 0.3,
                    'n_particles': len(endpoints)
                }
            raise ValueError("No particle endpoints available for origin estimation")
        
        pts = np.array([[e['lon'], e['lat']] for e in endpoints])
        
        # DBSCAN in approximate degrees (1 deg ≈ 111 km)
        eps_deg = 0.5  # ~55 km
        db = DBSCAN(eps=eps_deg, min_samples=5).fit(pts)
        labels = db.labels_
        
        # Largest non-noise cluster
        unique_labels, counts = np.unique(labels[labels >= 0], return_counts=True) if np.any(labels >= 0) else (np.array([]), np.array([]))
        
        if len(unique_labels) == 0:
            # All noise — use overall mean
            cluster_pts = pts
        else:
            best_label = unique_labels[np.argmax(counts)]
            cluster_pts = pts[labels == best_label]
        
        centroid_lon = float(np.mean(cluster_pts[:, 0]))
        centroid_lat = float(np.mean(cluster_pts[:, 1]))
        
        # Uncertainty = std deviation converted to km
        std_deg = float(np.std(np.linalg.norm(cluster_pts - [centroid_lon, centroid_lat], axis=1)))
        uncertainty_km = std_deg * 111.0
        
        confidence = min(0.95, len(cluster_pts) / max(len(pts), 1))
        
        # Origin time: acquisition_time minus hindcast duration (approximated as 48h ago)
        origin_time = (datetime.now(timezone.utc) - timedelta(hours=48)).isoformat()
        
        logger.info(f"Origin estimated at ({centroid_lon:.4f}, {centroid_lat:.4f}), uncertainty={uncertainty_km:.1f}km, confidence={confidence:.2f}")
        
        return {
            'origin_lon': round(centroid_lon, 4),
            'origin_lat': round(centroid_lat, 4),
            'origin_time': origin_time,
            'uncertainty_km': round(uncertainty_km, 2),
            'confidence': round(confidence, 3),
            'n_particles': len(cluster_pts)
        }
