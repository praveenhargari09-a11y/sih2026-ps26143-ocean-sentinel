import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
from loguru import logger
import numpy as np


# ── Module-level land polygon cache ─────────────────────────────────────────

_land_gdf_cache: gpd.GeoDataFrame | None = None
_land_gdf_loaded: bool = False


def _get_land_polygons() -> gpd.GeoDataFrame | None:
    """Load Natural Earth 110m land polygons once and cache."""
    global _land_gdf_cache, _land_gdf_loaded
    if _land_gdf_loaded:
        return _land_gdf_cache

    _land_gdf_loaded = True
    import json
    from pathlib import Path
    from shapely.geometry import shape

    ne_path = Path(__file__).resolve().parents[2] / "ne_110m.json"
    if not ne_path.is_file():
        logger.debug(f"Natural Earth land file not found at {ne_path}")
        return None

    try:
        with open(ne_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        geometries = [shape(feat["geometry"]) for feat in data.get("features", [])]
        _land_gdf_cache = gpd.GeoDataFrame(geometry=geometries, crs="EPSG:4326")
        logger.info(f"Loaded {len(geometries)} land polygons from {ne_path.name}")
    except Exception as e:
        logger.warning(f"Failed to load land polygons: {e}")
        _land_gdf_cache = None

    return _land_gdf_cache


class AISLoader:
    def load_csv(self, path: str) -> gpd.GeoDataFrame:
        logger.info(f"Loading AIS CSV from {path}")
        df = pd.read_csv(path)
        
        # Schema Normalization Mapping
        SCHEMA_MAP = {
            "latitude": "LAT",
            "longitude": "LON",
            "base_date_time": "BaseDateTime",
            "mmsi": "MMSI",
            "sog": "SOG",
            "cog": "COG",
            "heading": "Heading",
            "vessel_name": "VesselName",
            "vessel_type": "VesselType",
            "status": "Status",
            "imo": "IMO",
            "call_sign": "CallSign",
            "flag": "Flag"
        }
        
        # Normalize columns (case-insensitive match for incoming keys)
        incoming_cols = {col.lower(): col for col in df.columns}
        rename_dict = {}
        for expected_lower, internal_name in SCHEMA_MAP.items():
            if expected_lower in incoming_cols:
                rename_dict[incoming_cols[expected_lower]] = internal_name
            else:
                logger.warning(f"Expected AIS column '{expected_lower}' missing from input file.")
                
        df = df.rename(columns=rename_dict)
        
        # Ensure optional columns exist
        if 'Flag' not in df.columns:
            df['Flag'] = 'Unknown'
            
        df['BaseDateTime'] = pd.to_datetime(df['BaseDateTime'], errors='coerce')
        df = df.dropna(subset=['LAT', 'LON', 'BaseDateTime'])
        
        geometry = [Point(xy) for xy in zip(df['LON'], df['LAT'])]
        gdf = gpd.GeoDataFrame(df, geometry=geometry, crs="EPSG:4326")
        return gdf

    def clean(self, gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
        logger.info(f"Cleaning AIS data, initial rows: {len(gdf)}")
        gdf = gdf.dropna(subset=['LAT', 'LON'])
        gdf = gdf[(gdf['LAT'] >= -90) & (gdf['LAT'] <= 90)]
        gdf = gdf[(gdf['LON'] >= -180) & (gdf['LON'] <= 180)]
        
        if 'SOG' in gdf.columns:
            gdf = gdf[gdf['SOG'] <= 50]
            
        gdf = gdf.drop_duplicates(subset=['MMSI', 'BaseDateTime'], keep='first')
        logger.info(f"Cleaned AIS data, remaining rows: {len(gdf)}")

        # Land-mask validation: flag and remove AIS points on land
        gdf = self._filter_land_points(gdf)

        return gdf

    def _filter_land_points(self, gdf: gpd.GeoDataFrame,
                            tolerance_m: float = 500.0) -> gpd.GeoDataFrame:
        """
        Remove AIS points that fall on land using Natural Earth 110m polygons.
        Points within `tolerance_m` metres of the coastline are kept (port/dock
        tolerance for legitimate near-shore positions).
        """
        land_gdf = _get_land_polygons()
        if land_gdf is None:
            logger.debug("Land polygon data not available — skipping land-mask filter")
            return gdf

        initial_count = len(gdf)

        # Spatial join to find points that intersect land polygons
        on_land_mask = gdf.geometry.within(land_gdf.unary_union)

        if not on_land_mask.any():
            logger.info("Land-mask check passed — no AIS points on land")
            return gdf

        # For points flagged as on-land, check if they are within tolerance
        # distance of the coastline boundary (port/dock allowance)
        land_boundary = land_gdf.unary_union.boundary
        on_land_gdf = gdf[on_land_mask].copy()

        # Distance in degrees; tolerance_m → approximate degrees
        # 1° ≈ 111,000 m at equator, shrinks with cos(lat)
        mean_lat = on_land_gdf['LAT'].mean() if len(on_land_gdf) > 0 else 0
        deg_per_m = 1.0 / (111_000 * np.cos(np.radians(abs(mean_lat))))
        tol_deg = tolerance_m * deg_per_m

        near_coast_mask = on_land_gdf.geometry.distance(land_boundary) <= tol_deg

        # Points on land AND far from coast → remove
        far_inland = on_land_gdf[~near_coast_mask]
        near_coast = on_land_gdf[near_coast_mask]

        if len(far_inland) > 0:
            for _, row in far_inland.iterrows():
                mmsi = row.get('MMSI', '?')
                logger.warning(
                    f"LAND-MASK: Removing AIS point on land — MMSI={mmsi} "
                    f"lat={row['LAT']:.6f} lon={row['LON']:.6f} "
                    f"time={row.get('BaseDateTime', '?')}"
                )

            gdf = gdf[~gdf.index.isin(far_inland.index)]
            logger.info(
                f"Land-mask filter removed {len(far_inland)} inland point(s), "
                f"kept {len(near_coast)} near-coast point(s), "
                f"{len(gdf)}/{initial_count} rows remaining"
            )
        else:
            logger.info(
                f"Land-mask: {len(near_coast)} point(s) on land but within "
                f"{tolerance_m}m coast tolerance — all kept"
            )

        return gdf

    def interpolate_track(self, vessel_gdf: gpd.GeoDataFrame, freq: str = '5min') -> gpd.GeoDataFrame:
        # resample track to regular time intervals using linear interpolation on LAT/LON
        if vessel_gdf.empty:
            return vessel_gdf
            
        vessel_gdf = vessel_gdf.sort_values('BaseDateTime').set_index('BaseDateTime')
        # Only numeric interpolation
        numeric_cols = vessel_gdf.select_dtypes(include=[np.number]).columns.tolist()
        if 'LAT' not in numeric_cols:
            numeric_cols.append('LAT')
        if 'LON' not in numeric_cols:
            numeric_cols.append('LON')
            
        resampled = vessel_gdf[numeric_cols].resample(freq).mean().interpolate(method='linear')
        
        # Fill non-numeric with ffill
        non_numeric = vessel_gdf.select_dtypes(exclude=[np.number]).resample(freq).ffill()
        
        merged = pd.concat([resampled, non_numeric], axis=1).reset_index()
        geometry = [Point(xy) for xy in zip(merged['LON'], merged['LAT'])]
        
        return gpd.GeoDataFrame(merged, geometry=geometry, crs="EPSG:4326")
        
    def get_vessel_tracks(self, gdf: gpd.GeoDataFrame) -> dict:
        tracks = {}
        for mmsi, group in gdf.groupby('MMSI'):
            tracks[str(mmsi)] = group.copy().sort_values('BaseDateTime')
        return tracks
        
    def to_geojson_feature(self, vessel_gdf: gpd.GeoDataFrame, mmsi: str) -> dict:
        vessel_gdf = vessel_gdf.sort_values('BaseDateTime')
        coords = vessel_gdf[['LON', 'LAT']].values.tolist()
        properties = {
            'mmsi': str(mmsi)
        }
        for col in ['VesselName', 'VesselType', 'Flag']:
            if col in vessel_gdf.columns:
                properties[col] = str(vessel_gdf.iloc[0][col])
                
        return {
            'type': 'Feature',
            'properties': properties,
            'geometry': {
                'type': 'LineString',
                'coordinates': coords
            }
        }
