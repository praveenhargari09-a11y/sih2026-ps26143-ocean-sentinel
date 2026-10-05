import pandas as pd
import geopandas as gpd
from loguru import logger
import numpy as np
from sklearn.ensemble import IsolationForest

class AnomalyDetector:
    def detect_ais_darkness(self, track: gpd.GeoDataFrame, max_gap_minutes: float = 30) -> list:
        track = track.sort_values('BaseDateTime')
        if len(track) < 2:
            return []
            
        track['time_diff'] = track['BaseDateTime'].diff().dt.total_seconds() / 60.0
        gaps = track[track['time_diff'] > max_gap_minutes]
        
        darkness_events = []
        for idx, row in gaps.iterrows():
            prev_idx = track.index.get_loc(idx) - 1
            if prev_idx >= 0:
                prev_row = track.iloc[prev_idx]
                darkness_events.append({
                    'start': prev_row['BaseDateTime'].isoformat(),
                    'end': row['BaseDateTime'].isoformat(),
                    'duration_hours': row['time_diff'] / 60.0,
                    'lon_before': float(prev_row['LON']),
                    'lat_before': float(prev_row['LAT']),
                    'lon_after': float(row['LON']),
                    'lat_after': float(row['LAT'])
                })
        return darkness_events

    def detect_speed_anomaly(self, track: gpd.GeoDataFrame) -> dict:
        if 'SOG' not in track.columns or len(track) < 10:
            return {'is_anomaly': False, 'anomaly_score': 0.0, 'anomaly_timestamps': []}
            
        sog_data = track[['SOG']].fillna(track['SOG'].mean()).values
        clf = IsolationForest(contamination=0.05, random_state=42)
        preds = clf.fit_predict(sog_data)
        
        anomalies = track[preds == -1]
        is_anomaly = len(anomalies) > 0
        anomaly_score = float(len(anomalies) / len(track)) if len(track) > 0 else 0.0
        
        timestamps = anomalies['BaseDateTime'].dt.strftime('%Y-%m-%dT%H:%M:%SZ').tolist()
        
        return {
            'is_anomaly': is_anomaly,
            'anomaly_score': anomaly_score,
            'anomaly_timestamps': timestamps
        }

    def detect_heading_anomaly(self, track: gpd.GeoDataFrame) -> dict:
        if 'COG' not in track.columns or len(track) < 2:
            return {'is_anomaly': False, 'anomaly_timestamps': []}
            
        track = track.sort_values('BaseDateTime')
        track['cog_diff'] = track['COG'].diff().abs()
        # Handle 360 wrap around
        track['cog_diff'] = np.minimum(track['cog_diff'], 360 - track['cog_diff'])
        track['time_diff'] = track['BaseDateTime'].diff().dt.total_seconds() / 60.0
        
        anomalies = track[(track['cog_diff'] > 45) & (track['time_diff'] <= 10)]
        is_anomaly = len(anomalies) > 0
        timestamps = anomalies['BaseDateTime'].dt.strftime('%Y-%m-%dT%H:%M:%SZ').tolist()
        
        return {
            'is_anomaly': is_anomaly,
            'anomaly_timestamps': timestamps
        }
        
    def score_anomalies(self, darkness: list, speed: dict, heading: dict) -> float:
        darkness_count = len(darkness)
        d_score = min(1.0, darkness_count * 0.5) # 2 gaps = max score
        s_score = 1.0 if speed.get('is_anomaly', False) else 0.0
        h_score = 1.0 if heading.get('is_anomaly', False) else 0.0
        
        total = (d_score * 0.5) + (s_score * 0.3) + (h_score * 0.2)
        return total
