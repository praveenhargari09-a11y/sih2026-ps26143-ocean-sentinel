import geopandas as gpd
from loguru import logger
from typing import List, Dict
from .anomaly_detector import AnomalyDetector
from .scoring import VesselScorer
from .ais_loader import AISLoader

class VesselRanker:
    def __init__(self):
        self.detector = AnomalyDetector()
        self.scorer = VesselScorer()
        
    def rank(self, candidate_vessels: dict, spill_info: dict, origin: dict) -> List[dict]:
        results = []
        origin_lon = origin.get('origin_lon', 0.0)
        origin_lat = origin.get('origin_lat', 0.0)
        origin_time = origin.get('origin_time', '')
        spill_elongation = spill_info.get('elongation_deg', 0.0)
        
        for mmsi, track_gdf in candidate_vessels.items():
            if track_gdf.empty:
                continue
                
            darkness = self.detector.detect_ais_darkness(track_gdf)
            speed_anom = self.detector.detect_speed_anomaly(track_gdf)
            head_anom = self.detector.detect_heading_anomaly(track_gdf)
            
            vtype = str(track_gdf.iloc[0].get('VesselType', 'Unknown'))
            vname = str(track_gdf.iloc[0].get('VesselName', f"Vessel_{mmsi}"))
            flag = str(track_gdf.iloc[0].get('Flag', 'Unknown'))
            
            scores = {
                'proximity': self.scorer.score_proximity(track_gdf, origin_lon, origin_lat, origin_time),
                'trajectory': self.scorer.score_trajectory_alignment(track_gdf, spill_elongation),
                'vessel_type': self.scorer.score_vessel_type(vtype),
                'darkness': self.scorer.score_ais_darkness(darkness, origin_time, origin_lon, origin_lat),
                'speed': self.scorer.score_speed_anomaly(speed_anom)
            }
            
            total_score = self.scorer.compute_total(scores)
            
            # Compute min proximity km exactly for reporting
            distances = track_gdf.apply(lambda r: self.scorer.haversine(r['LON'], r['LAT'], origin_lon, origin_lat), axis=1)
            min_proximity_km = distances.min() if not distances.empty else 9999.0
            
            # Basic GeoJSON extraction without relying on ais_loader instantiated here
            track_gdf_sorted = track_gdf.sort_values('BaseDateTime')
            coords = track_gdf_sorted[['LON', 'LAT']].values.tolist()
            track_geojson = {
                'type': 'Feature',
                'properties': {'mmsi': mmsi, 'VesselName': vname, 'VesselType': vtype},
                'geometry': {'type': 'LineString', 'coordinates': coords}
            }
            
            results.append({
                'mmsi': mmsi,
                'vessel_name': vname,
                'vessel_type': vtype,
                'flag': flag,
                'total_score': round(total_score, 4),
                'score_breakdown': {k: round(v, 4) for k, v in scores.items()},
                'proximity_km': round(min_proximity_km, 2),
                'ais_gaps': darkness,
                'track_geojson': track_geojson
            })
            
        results.sort(key=lambda x: x['total_score'], reverse=True)
        for i, res in enumerate(results):
            res['rank'] = i + 1
            
        return results

def rank_from_csv(ais_csv_path: str, spill_info: dict, origin: dict) -> List[dict]:
    loader = AISLoader()
    raw_gdf = loader.load_csv(ais_csv_path)
    clean_gdf = loader.clean(raw_gdf)
    tracks = loader.get_vessel_tracks(clean_gdf)
    
    ranker = VesselRanker()
    return ranker.rank(tracks, spill_info, origin)
