import geopandas as gpd
from shapely.geometry import Point
from datetime import datetime, timedelta, timezone
from loguru import logger
import pandas as pd

class SpatialFilter:
    def filter_by_origin(self, ais_gdf: gpd.GeoDataFrame, origin_lon: float, origin_lat: float, origin_time_utc: str, search_radius_km: float = 100, time_window_hours: float = 48) -> dict:
        logger.info(f"Filtering AIS data spatially around ({origin_lon}, {origin_lat}) radius {search_radius_km}km")
        
        radius_deg = search_radius_km / 111.0
        origin_point = Point(origin_lon, origin_lat)
        buffer = origin_point.buffer(radius_deg)
        
        if isinstance(origin_time_utc, str):
            origin_time = datetime.fromisoformat(origin_time_utc.replace('Z', '+00:00'))
        else:
            origin_time = origin_time_utc
            
        start_time = origin_time - timedelta(hours=time_window_hours)
        end_time = origin_time + timedelta(hours=6)
        
        if ais_gdf['BaseDateTime'].dt.tz is None:
            # Assume UTC
            ais_gdf['BaseDateTime'] = ais_gdf['BaseDateTime'].dt.tz_localize('UTC')
        else:
            ais_gdf['BaseDateTime'] = ais_gdf['BaseDateTime'].dt.tz_convert('UTC')
            
        if start_time.tzinfo is None:
            start_time = start_time.replace(tzinfo=timezone.utc)
        if end_time.tzinfo is None:
            end_time = end_time.replace(tzinfo=timezone.utc)
            
        time_mask = (ais_gdf['BaseDateTime'] >= start_time) & (ais_gdf['BaseDateTime'] <= end_time)
        filtered = ais_gdf[time_mask].copy()
        
        # Spatial intersection
        # using the spatial index
        intersect_mask = filtered.geometry.intersects(buffer)
        spatial_filtered = filtered[intersect_mask]
        
        candidate_mmsis = spatial_filtered['MMSI'].unique()
        logger.info(f"Found {len(candidate_mmsis)} candidate vessels in spatial/temporal filter")
        
        results = {}
        for mmsi in candidate_mmsis:
            # Return full track for candidate vessels, not just the filtered part
            vessel_track = ais_gdf[ais_gdf['MMSI'] == mmsi].copy().sort_values('BaseDateTime')
            results[str(mmsi)] = vessel_track
            
        return results
