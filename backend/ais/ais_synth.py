import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
from loguru import logger
import numpy as np
import math
import yaml
from datetime import datetime, timedelta, timezone

class SyntheticAISGenerator:
    def generate(self, config_path: str, output_path: str = None) -> gpd.GeoDataFrame:
        logger.info(f"Generating synthetic AIS data based on {config_path}")
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        # Support both flat config and nested scenario_config.yaml
        spill_cfg = config.get('spill', config)
        spill_lon = config.get('origin_lon', spill_cfg.get('centroid_lon', 0.0))
        spill_lat = config.get('origin_lat', spill_cfg.get('centroid_lat', 0.0))
        start_time_str = config.get('start_time', spill_cfg.get('acquisition_time', datetime.now(timezone.utc).isoformat()))
        start_time = datetime.fromisoformat(start_time_str.replace('Z', '+00:00'))
        
        vessels = config.get('vessels', [])
        if not vessels and 'ais' in config:
            vessels = config['ais'].get('vessels', [])
        all_records = []
        
        for v in vessels:
            mmsi = str(v.get('mmsi', np.random.randint(100000000, 999999999)))
            is_guilty = v.get('is_guilty', v.get('guilty', False))
            
            if is_guilty:
                # Passes exactly through spill centroid
                angle = math.radians(v.get('heading', 45))
                speed_knots = v.get('speed', 12)
                speed_deg_hr = speed_knots * 0.017 # approx 1 knot ~ 0.017 deg/hr
                
                # Start 6 hours before origin time (keeps track within ~120km of spill)
                dt = start_time - timedelta(hours=6)
                lon_start = spill_lon - (math.sin(angle) * speed_deg_hr * 6)
                lat_start = spill_lat - (math.cos(angle) * speed_deg_hr * 6)
                
                records = self._generate_linear_track(mmsi, v, dt, 12, lon_start, lat_start, angle, speed_knots)
                
                # Add gap
                gap_start_str = v.get('gap_start', v.get('ais_gap_start', (start_time - timedelta(hours=2)).isoformat()))
                gap_hours = v.get('gap_hours', v.get('ais_gap_hours', 4))
                df_track = pd.DataFrame(records)
                df_track = self.add_ais_darkness(df_track, mmsi, gap_start_str, gap_hours)
                all_records.extend(df_track.to_dict('records'))
                
            else:
                # Random linear track near the spill area
                angle = math.radians(np.random.uniform(0, 360))
                speed_knots = np.random.uniform(8, 15)
                lon_start = spill_lon + np.random.uniform(0.1, 0.5) * np.random.choice([-1, 1])
                lat_start = spill_lat + np.random.uniform(0.1, 0.5) * np.random.choice([-1, 1])
                dt = start_time - timedelta(hours=6)
                
                records = self._generate_linear_track(mmsi, v, dt, 12, lon_start, lat_start, angle, speed_knots)
                all_records.extend(records)
                
        df = pd.DataFrame(all_records)
        geometry = [Point(xy) for xy in zip(df['LON'], df['LAT'])]
        gdf = gpd.GeoDataFrame(df, geometry=geometry, crs="EPSG:4326")
        
        if output_path:
            gdf.drop(columns=['geometry']).to_csv(output_path, index=False)
            
        return gdf
        
    def _generate_linear_track(self, mmsi, v_info, start_time, duration_hours, lon_start, lat_start, angle, speed_knots):
        records = []
        speed_deg_hr = speed_knots * 0.017
        steps = int(duration_hours * 60 / 5) # 5 min intervals
        
        curr_lon = lon_start
        curr_lat = lat_start
        
        for i in range(steps):
            t = start_time + timedelta(minutes=i*5)
            # Add slight random drift
            angle += math.radians(np.random.normal(0, 0.5))
            
            curr_lon += (math.sin(angle) * speed_deg_hr * (5/60.0))
            curr_lat += (math.cos(angle) * speed_deg_hr * (5/60.0))
            
            records.append({
                'MMSI': mmsi,
                'BaseDateTime': t,
                'LAT': curr_lat,
                'LON': curr_lon,
                'SOG': speed_knots + np.random.normal(0, 0.5),
                'COG': math.degrees(angle) % 360,
                'Heading': math.degrees(angle) % 360,
                'VesselName': v_info.get('name', f"Vessel_{mmsi}"),
                'VesselType': v_info.get('vessel_type', v_info.get('type', 'Cargo')),
                'Flag': v_info.get('flag', 'Unknown'),
                'IMO': v_info.get('imo', ''),
                'Status': 0
            })
        return records

    def add_ais_darkness(self, df: pd.DataFrame, mmsi: str, gap_start_str: str, gap_hours: float) -> pd.DataFrame:
        gap_start = datetime.fromisoformat(gap_start_str.replace('Z', '+00:00'))
        gap_end = gap_start + timedelta(hours=gap_hours)
        mask = (df['MMSI'] == mmsi) & (df['BaseDateTime'] >= gap_start) & (df['BaseDateTime'] <= gap_end)
        return df[~mask].copy()
        
def generate_demo_ais(config_path: str, output_path: str = None) -> gpd.GeoDataFrame:
    gen = SyntheticAISGenerator()
    return gen.generate(config_path, output_path)
