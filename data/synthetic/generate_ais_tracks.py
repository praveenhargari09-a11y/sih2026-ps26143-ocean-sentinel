"""
Generate synthetic AIS vessel tracks for oil spill attribution demonstration.
"""
import argparse, math, yaml
from datetime import datetime, timedelta, timezone
from pathlib import Path
import numpy as np
import pandas as pd
from loguru import logger

try:
    from geopandas import GeoDataFrame
    from shapely.geometry import Point
    HAS_GPD = True
except ImportError:
    HAS_GPD = False

class SyntheticAISGenerator:
    def generate(self, config_path: str, output_path: str = None):
        with open(config_path) as f:
            cfg = yaml.safe_load(f)
        
        spill = cfg['spill']
        ais_cfg = cfg['ais']
        origin_lon = spill['centroid_lon']
        origin_lat = spill['centroid_lat']
        acq_time = datetime.fromisoformat(spill['acquisition_time'].replace('Z', '+00:00'))
        start_time = acq_time - timedelta(hours=48)
        end_time = acq_time + timedelta(hours=24)
        
        rng = np.random.default_rng(99)
        all_records = []
        
        for vessel in ais_cfg['vessels']:
            mmsi = vessel['mmsi']
            name = vessel['name']
            vtype = vessel.get('vessel_type', 'Cargo')
            flag = vessel.get('flag', 'Unknown')
            imo = str(rng.integers(1000000, 9999999))
            guilty = vessel.get('guilty', False)
            
            if guilty:
                # Guilty vessel: track through origin point
                # Start 2 degrees away, end 2 degrees away on other side
                start_lon = origin_lon - 2.0 + rng.uniform(-0.2, 0.2)
                start_lat = origin_lat - 1.5 + rng.uniform(-0.2, 0.2)
                end_lon = origin_lon + 2.0 + rng.uniform(-0.2, 0.2)
                end_lat = origin_lat + 1.0 + rng.uniform(-0.2, 0.2)
                speed_kt = rng.uniform(10, 14)  # typical tanker speed
            else:
                # Innocent: random direction, not through origin
                offset_lon = rng.uniform(-2.0, 2.0)
                offset_lat = rng.uniform(-2.0, 2.0)
                # Ensure it doesn't pass exactly through origin
                if abs(offset_lon) < 0.3: offset_lon = 0.5 * np.sign(offset_lon) if offset_lon != 0 else 0.5
                start_lon = origin_lon + offset_lon
                start_lat = origin_lat + offset_lat
                heading_deg = rng.uniform(0, 360)
                dist_deg = rng.uniform(1.0, 3.0)
                end_lon = start_lon + dist_deg * math.sin(math.radians(heading_deg))
                end_lat = start_lat + dist_deg * math.cos(math.radians(heading_deg))
                speed_kt = rng.uniform(8, 15)
            
            current_time = start_time
            current_lon = start_lon
            current_lat = start_lat
            
            total_hours = (end_time - start_time).total_seconds() / 3600
            dlon_per_hr = (end_lon - start_lon) / total_hours
            dlat_per_hr = (end_lat - start_lat) / total_hours
            
            gap_start = None
            gap_end = None
            if guilty and vessel.get('ais_gap_start'):
                gap_start = datetime.fromisoformat(vessel['ais_gap_start'].replace('Z', '+00:00'))
                gap_end = gap_start + timedelta(hours=vessel.get('ais_gap_hours', 6))
            
            dt_minutes = 5
            dt_hours = dt_minutes / 60.0
            
            while current_time <= end_time:
                # Skip during AIS gap
                if gap_start and gap_start <= current_time <= gap_end:
                    current_time += timedelta(minutes=dt_minutes)
                    current_lon += dlon_per_hr * dt_hours + rng.normal(0, 0.001)
                    current_lat += dlat_per_hr * dt_hours + rng.normal(0, 0.001)
                    continue
                
                # Add small heading drift
                cog = math.degrees(math.atan2(dlon_per_hr, dlat_per_hr)) % 360
                cog += rng.normal(0, 2)  # slight variation
                
                all_records.append({
                    'MMSI': mmsi,
                    'BaseDateTime': current_time.strftime('%Y-%m-%dT%H:%M:%S'),
                    'LAT': round(current_lat + rng.normal(0, 0.0005), 6),
                    'LON': round(current_lon + rng.normal(0, 0.0005), 6),
                    'SOG': round(float(speed_kt) + rng.normal(0, 0.5), 1),
                    'COG': round(float(cog) % 360, 1),
                    'Heading': round(float(cog) % 360, 0),
                    'VesselName': name,
                    'VesselType': vtype,
                    'Flag': flag,
                    'IMO': imo,
                    'Status': '0',
                })
                current_lon += dlon_per_hr * dt_hours + rng.normal(0, 0.001)
                current_lat += dlat_per_hr * dt_hours + rng.normal(0, 0.001)
                current_time += timedelta(minutes=dt_minutes)
        
        df = pd.DataFrame(all_records)
        logger.info(f'Generated {len(df)} AIS records for {len(ais_cfg["vessels"])} vessels')
        
        if output_path:
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            df.to_csv(output_path, index=False)
            logger.info(f'Saved AIS tracks to {output_path}')
        
        if HAS_GPD:
            geometry = [Point(row.LON, row.LAT) for _, row in df.iterrows()]
            gdf = GeoDataFrame(df, geometry=geometry, crs='EPSG:4326')
            return gdf
        return df
    
    def add_ais_darkness(self, df: pd.DataFrame, mmsi: str, gap_start_str: str, gap_hours: float) -> pd.DataFrame:
        gap_start = pd.to_datetime(gap_start_str)
        gap_end = gap_start + timedelta(hours=gap_hours)
        mask = (df['MMSI'] == mmsi) & \
               (pd.to_datetime(df['BaseDateTime']) >= gap_start) & \
               (pd.to_datetime(df['BaseDateTime']) <= gap_end)
        return df[~mask].reset_index(drop=True)

def generate_demo_ais(config_path: str, output_path: str = None):
    return SyntheticAISGenerator().generate(config_path, output_path)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='data/synthetic/scenario_config.yaml')
    parser.add_argument('--output', default='data/raw/synthetic_ais.csv')
    args = parser.parse_args()
    gen = SyntheticAISGenerator()
    result = gen.generate(args.config, args.output)
    print(f'Generated {len(result)} AIS records')
