import pandas as pd
import geopandas as gpd
from loguru import logger
import math
from datetime import datetime, timedelta, timezone

class VesselScorer:
    WEIGHTS = {
        'proximity': 0.35, 
        'trajectory': 0.25, 
        'darkness': 0.20, 
        'vessel_type': 0.10, 
        'speed': 0.10
    }

    def haversine(self, lon1: float, lat1: float, lon2: float, lat2: float) -> float:
        R = 6371.0 # km
        dlat = math.radians(lat2 - lat1)
        dlon = math.radians(lon2 - lon1)
        a = math.sin(dlat/2)**2 + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dlon/2)**2
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
        return R * c

    def score_proximity(self, track: gpd.GeoDataFrame, origin_lon: float, origin_lat: float, origin_time: str, window_hours: float = 6) -> float:
        ot = datetime.fromisoformat(origin_time.replace('Z', '+00:00'))
        if ot.tzinfo is None:
            ot = ot.replace(tzinfo=timezone.utc)
            
        start = ot - timedelta(hours=window_hours)
        end = ot + timedelta(hours=1)
        
        t_track = track.copy()
        if t_track['BaseDateTime'].dt.tz is None:
            t_track['BaseDateTime'] = t_track['BaseDateTime'].dt.tz_localize('UTC')
        else:
            t_track['BaseDateTime'] = t_track['BaseDateTime'].dt.tz_convert('UTC')
            
        mask = (t_track['BaseDateTime'] >= start) & (t_track['BaseDateTime'] <= end)
        subset = t_track[mask]
        
        if subset.empty:
            # Gap-aware fallback: use last-known position BEFORE the scoring window.
            # This handles vessels that went dark during the spill time — their
            # last-known position is a reasonable proxy for where they were.
            before_mask = t_track['BaseDateTime'] < start
            before_pts = t_track[before_mask].sort_values('BaseDateTime')
            
            if not before_pts.empty:
                last = before_pts.iloc[-1]
                dist_km = self.haversine(last['LON'], last['LAT'], origin_lon, origin_lat)
                # Apply a 0.8 ceiling — gap-based proximity is inherently less
                # certain than direct observation, so cap below 1.0
                score = max(0.0, 1.0 - (dist_km / 200.0))
                logger.debug(
                    f"Proximity gap-fallback: last-known {dist_km:.1f} km from origin "
                    f"-> score {score:.3f} (capped at 0.8)"
                )
                return min(score, 0.8)
            
            return 0.0
            
        distances = subset.apply(lambda r: self.haversine(r['LON'], r['LAT'], origin_lon, origin_lat), axis=1)
        min_dist_km = distances.min()
        
        return max(0.0, 1.0 - (min_dist_km / 200.0))

    def score_trajectory_alignment(self, track: gpd.GeoDataFrame, spill_elongation_deg: float) -> float:
        if 'COG' not in track.columns or len(track) == 0:
            return 0.5
            
        mean_cog = track['COG'].mean()
        # Difference modulo 180 since elongation is bidirectional
        diff = abs((mean_cog - spill_elongation_deg) % 180)
        diff = min(diff, 180 - diff) # shortest angular diff up to 90
        
        return max(0.0, 1.0 - (diff / 90.0))

    def score_vessel_type(self, vessel_type: str) -> float:
        vt = str(vessel_type).lower()
        if 'tanker' in vt:
            return 1.0
        elif 'bulk' in vt:
            return 0.8
        elif 'general cargo' in vt:
            return 0.55
        elif 'cargo' in vt:
            return 0.6
        elif 'container' in vt:
            return 0.5
        return 0.3

    def score_ais_darkness(self, darkness_events: list, origin_time: str, origin_lon: float, origin_lat: float, radius_km: float = 50) -> float:
        if not darkness_events:
            return 0.0
            
        ot = datetime.fromisoformat(origin_time.replace('Z', '+00:00'))
        if ot.tzinfo is None:
            ot = ot.replace(tzinfo=timezone.utc)
            
        max_score = 0.0
        for ev in darkness_events:
            start_t = datetime.fromisoformat(ev['start'].replace('Z', '+00:00'))
            end_t = datetime.fromisoformat(ev['end'].replace('Z', '+00:00'))
            
            if start_t.tzinfo is None: start_t = start_t.replace(tzinfo=timezone.utc)
            if end_t.tzinfo is None: end_t = end_t.replace(tzinfo=timezone.utc)
            
            time_overlap = (start_t <= ot <= end_t)
            
            dist = self.haversine(ev['lon_before'], ev['lat_before'], origin_lon, origin_lat)
            spatial_overlap = dist <= radius_km
            
            if time_overlap and spatial_overlap:
                max_score = max(max_score, 1.0)
            elif time_overlap:
                max_score = max(max_score, 0.5)
            elif spatial_overlap:
                max_score = max(max_score, 0.3)
                
        return max_score

    def score_speed_anomaly(self, speed_anomaly: dict) -> float:
        if speed_anomaly.get('is_anomaly', False):
            return speed_anomaly.get('anomaly_score', 0.5)
        return 0.1

    def compute_total(self, scores: dict) -> float:
        total = 0.0
        weight_sum = 0.0
        for k, v in self.WEIGHTS.items():
            total += scores.get(k, 0.0) * v
            weight_sum += v
        return total / weight_sum if weight_sum > 0 else 0.0
