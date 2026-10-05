"""
backend/drift/drift_engine.py
===============================
Core Lagrangian Drift Engine.

Supports:
  1. OpenDrift (full physics model) — when opendrift is installed.
  2. Synthetic Lagrangian fallback — pure Python, no external dependencies.
     Simulates drift using vector advection + random walk.

The fallback model is intentionally simple but physically meaningful enough
for demonstration, testing, and environments without OpenDrift.

Drift physics (fallback):
  dx/dt = u_current + α·u_wind    (Stokes drift factor α ≈ 0.03)
  dy/dt = v_current + α·v_wind
  + random walk: σ ≈ 0.1 km/√h (turbulent diffusion)
"""

from __future__ import annotations

import math
import os
from datetime import datetime, timedelta, timezone
from typing import Callable, Optional

import numpy as np
from loguru import logger

# Optional OpenDrift import
try:
    from opendrift.models.openoil import OpenOil  # type: ignore
    from opendrift.readers import reader_netCDF_CF_generic  # type: ignore
    OPENDRIFT_AVAILABLE = True
    logger.info("OpenDrift detected — full physics drift model enabled.")
except ImportError:
    OPENDRIFT_AVAILABLE = False
    logger.warning(
        "OpenDrift not installed. Using synthetic Lagrangian fallback. "
        "Install with: pip install opendrift"
    )

# Number of seconds per degree of latitude (approx)
SEC_PER_DEG_LAT = 111_320.0  # 1° lat ≈ 111.32 km
# Stokes drift factor: fraction of wind speed added to particle velocity
STOKES_FACTOR = 0.03


def _extract_times(times_raw, n_times: int) -> list[datetime]:
    """
    Convert OpenDrift time output to a plain list of datetime objects.

    OpenDrift's get_property("time") can return xarray DataArrays,
    numpy datetime64 arrays, or tuples. This normalises them.
    """
    import pandas as pd

    # If it's a tuple/list of xarray arrays, take the first element
    if isinstance(times_raw, (tuple, list)):
        times_raw = times_raw[0]

    # xarray DataArray → numpy
    if hasattr(times_raw, 'values'):
        times_raw = times_raw.values

    # numpy datetime64 array
    if hasattr(times_raw, 'dtype') and np.issubdtype(getattr(times_raw, 'dtype', None), np.datetime64):
        arr = np.asarray(times_raw).ravel()
        # Take only the first n_times entries (time dimension, not trajectory)
        arr = arr[:n_times]
        return [pd.Timestamp(t).to_pydatetime().replace(tzinfo=None) for t in arr]

    # Already a list of datetimes
    if isinstance(times_raw, (list, np.ndarray)):
        result = []
        for t in times_raw:
            if isinstance(t, datetime):
                result.append(t)
            else:
                try:
                    result.append(pd.Timestamp(t).to_pydatetime().replace(tzinfo=None))
                except Exception:
                    result.append(datetime.now())
        return result[:n_times]

    # Fallback: generate hourly timestamps
    logger.warning("Could not parse OpenDrift times; generating synthetic timestamps.")
    return [datetime(2000, 1, 1) + timedelta(hours=i) for i in range(n_times)]


class DriftEngine:
    """
    Lagrangian particle drift engine for oil spill trajectory simulation.

    Parameters
    ----------
    ocean_data_path : str
        Path to NetCDF file with ocean current fields (``uo``, ``vo``).
    wind_data_path : str
        Path to NetCDF file with wind fields (``u10``, ``v10``).
        Can be the same file as ``ocean_data_path`` if combined.
    """

    def __init__(self, ocean_data_path: str, wind_data_path: str) -> None:
        self.ocean_data_path = ocean_data_path
        self.wind_data_path = wind_data_path
        self._readers = None

        if OPENDRIFT_AVAILABLE:
            self._setup_readers()

    # ------------------------------------------------------------------
    # Reader setup
    # ------------------------------------------------------------------

    def _setup_readers(self) -> None:
        """
        Attempt to set up OpenDrift NetCDF readers.

        Falls back gracefully if the NetCDF files are not real forcing data.
        """
        mapping = {
            'uo': 'x_sea_water_velocity',
            'vo': 'y_sea_water_velocity',
            'u10': 'x_wind',
            'v10': 'y_wind',
        }
        try:
            self._ocean_reader = reader_netCDF_CF_generic.Reader(
                self.ocean_data_path, standard_name_mapping=mapping
            )
            logger.debug(f"Ocean reader loaded: {self.ocean_data_path}")
        except Exception as exc:
            logger.warning(f"Could not load ocean reader: {exc}")
            self._ocean_reader = None

        import netCDF4 as nc
        try:
            ds = nc.Dataset(self.wind_data_path)
            has_wind = 'u10' in ds.variables and 'v10' in ds.variables
            ds.close()
        except Exception:
            has_wind = False

        if has_wind:
            try:
                self._wind_reader = reader_netCDF_CF_generic.Reader(
                    self.wind_data_path, standard_name_mapping=mapping
                )
                logger.debug(f"Wind reader loaded: {self.wind_data_path}")
            except Exception as exc:
                logger.warning(f"Could not load wind reader: {exc}")
                self._wind_reader = None
        else:
            from opendrift.readers.reader_constant import Reader as ConstantReader
            # Default wind from Ennore incident.yaml if missing
            self._wind_reader = ConstantReader({'x_wind': -3.39, 'y_wind': -3.39})
            logger.debug("NetCDF lacks wind data. Using fallback ConstantReader for wind.")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run_hindcast(
        self,
        lon: float,
        lat: float,
        time_satellite: str,
        duration_hours: int = 48,
        n_particles: int = 100,
    ) -> dict:
        """
        Run a backward hindcast simulation from the observed spill location.

        Particles are seeded at the observed slick centroid and advected
        **backward** in time to find the probable release origin.

        Parameters
        ----------
        lon : float — observed spill centroid longitude (°E)
        lat : float — observed spill centroid latitude (°N)
        time_satellite : str — ISO 8601 satellite acquisition time
        duration_hours : int — length of hindcast (hours backward)
        n_particles : int — number of Lagrangian particles

        Returns
        -------
        dict — GeoJSON FeatureCollection of particle trajectories
        """
        logger.info(
            f"Running hindcast: origin=({lon:.4f},{lat:.4f}), "
            f"duration={duration_hours}h, n_particles={n_particles}"
        )

        if OPENDRIFT_AVAILABLE and self._ocean_reader is not None:
            result = self._opendrift_hindcast(lon, lat, time_satellite, duration_hours, n_particles)
        else:
            result = self._synthetic_drift(
                lon=lon,
                lat=lat,
                time_start=time_satellite,
                duration_hours=duration_hours,
                n_particles=n_particles,
                backward=True,
            )
        return result

    def run_forecast(
        self,
        lon: float,
        lat: float,
        time_satellite: str,
        duration_hours: int = 72,
        n_particles: int = 100,
    ) -> dict:
        """
        Run a forward forecast simulation from the observed spill location.

        Parameters
        ----------
        lon, lat : float
        time_satellite : str
        duration_hours : int
        n_particles : int

        Returns
        -------
        dict — GeoJSON FeatureCollection
        """
        logger.info(
            f"Running forecast: origin=({lon:.4f},{lat:.4f}), "
            f"duration={duration_hours}h, n_particles={n_particles}"
        )

        if OPENDRIFT_AVAILABLE and self._ocean_reader is not None:
            return self._opendrift_forecast(lon, lat, time_satellite, duration_hours, n_particles)
        else:
            return self._synthetic_drift(
                lon=lon,
                lat=lat,
                time_start=time_satellite,
                duration_hours=duration_hours,
                n_particles=n_particles,
                backward=False,
            )

    # ------------------------------------------------------------------
    # OpenDrift implementation
    # ------------------------------------------------------------------

    def _opendrift_hindcast(
        self, lon, lat, time_satellite, duration_hours, n_particles
    ) -> dict:
        """Run hindcast using OpenDrift's OpenOil model."""
        t_sat = datetime.fromisoformat(time_satellite.replace("Z", ""))
        if t_sat.tzinfo is not None:
            t_sat = t_sat.replace(tzinfo=None)
        
        t_start = t_sat
        t_end = t_sat - timedelta(hours=duration_hours)

        # Task 2b: Clamp the requested duration to what the grid supports
        clamped = False
        actual_duration = duration_hours
        if self._ocean_reader:
            r_start = self._ocean_reader.start_time
            if r_start.tzinfo is not None:
                r_start = r_start.replace(tzinfo=None)
            
            # If the start time itself is completely out of bounds, abort gracefully
            if t_start < r_start:
                logger.warning("Spill time is entirely before the NetCDF time coverage!")
                raise ValueError("Spill time is outside current data coverage")
                
            if t_end < r_start:
                # Clamp the end time to the reader's start time
                t_end = r_start
                actual_duration = (t_start - t_end).total_seconds() / 3600.0
                clamped = True
                logger.warning(f"Hindcast duration clamped from {duration_hours}h to {actual_duration:.1f}h to stay within data coverage.")

        try:
            o = OpenOil(loglevel=50)
            if self._ocean_reader:
                o.add_reader(self._ocean_reader)
            if self._wind_reader:
                o.add_reader(self._wind_reader)

            lons_seed = lon + np.random.normal(0, 0.01, n_particles)
            lats_seed = lat + np.random.normal(0, 0.01, n_particles)

            o.seed_elements(
                lon=lons_seed,
                lat=lats_seed,
                time=t_start,
                number=n_particles,
                oil_type="GENERIC INTERMEDIATE FUEL OIL 180",
            )

            o.run(
                end_time=t_end,
                time_step=timedelta(hours=-1),  # backward step
                time_step_output=timedelta(hours=1),
                outfile=None,
            )

            # OpenDrift get_property returns (n_times, n_trajectories);
            # transpose to (n_particles, n_times) for our GeoJSON builder.
            lons_raw = o.get_property("lon")[0]   # shape (n_times, n_trajectories)
            lats_raw = o.get_property("lat")[0]
            lons_out = np.asarray(lons_raw).T      # → (n_particles, n_times)
            lats_out = np.asarray(lats_raw).T

            # Extract proper datetime list from the time dimension
            times_raw = o.get_property("time")
            times_out = _extract_times(times_raw, lons_raw.shape[0])

            res = self._trajectories_to_geojson(lons_out, lats_out, times_out)
            if clamped:
                res["_clamped_warning"] = f"Hindcast limited to {actual_duration:.1f}h — current data coverage limit."
            return res
            
        except (Exception, SystemExit) as exc:
            logger.warning(f"OpenDrift hindcast aborted/failed: {exc}. Using synthetic fallback.")
            return self._synthetic_drift(lon, lat, time_satellite, duration_hours, n_particles, backward=True)

    def _opendrift_forecast(
        self, lon, lat, time_satellite, duration_hours, n_particles
    ) -> dict:
        """Run forecast using OpenDrift's OpenOil model."""
        t_sat = datetime.fromisoformat(time_satellite.replace("Z", ""))
        if t_sat.tzinfo is not None:
            t_sat = t_sat.replace(tzinfo=None)
            
        t_start = t_sat
        t_end = t_sat + timedelta(hours=duration_hours)

        clamped = False
        actual_duration = duration_hours
        if self._ocean_reader:
            r_end = self._ocean_reader.end_time
            if r_end.tzinfo is not None:
                r_end = r_end.replace(tzinfo=None)
                
            if t_start > r_end:
                logger.warning("Spill time is entirely after the NetCDF time coverage!")
                raise ValueError("Spill time is outside current data coverage")
                
            if t_end > r_end:
                t_end = r_end
                actual_duration = (t_end - t_start).total_seconds() / 3600.0
                clamped = True
                logger.warning(f"Forecast duration clamped from {duration_hours}h to {actual_duration:.1f}h to stay within data coverage.")

        try:
            o = OpenOil(loglevel=50)
            if self._ocean_reader:
                o.add_reader(self._ocean_reader)
            if self._wind_reader:
                o.add_reader(self._wind_reader)

            lons_seed = lon + np.random.normal(0, 0.01, n_particles)
            lats_seed = lat + np.random.normal(0, 0.01, n_particles)

            o.seed_elements(
                lon=lons_seed,
                lat=lats_seed,
                time=t_start,
                number=n_particles,
                oil_type="GENERIC INTERMEDIATE FUEL OIL 180",
            )

            o.run(
                end_time=t_end,
                time_step=timedelta(hours=1),
                time_step_output=timedelta(hours=1),
                outfile=None,
            )

            # Same transpose as hindcast — OpenDrift returns (n_times, n_trajectories)
            lons_raw = o.get_property("lon")[0]
            lats_raw = o.get_property("lat")[0]
            lons_out = np.asarray(lons_raw).T
            lats_out = np.asarray(lats_raw).T

            times_raw = o.get_property("time")
            times_out = _extract_times(times_raw, lons_raw.shape[0])

            res = self._trajectories_to_geojson(lons_out, lats_out, times_out)
            if clamped:
                res["_clamped_warning"] = f"Forecast limited to {actual_duration:.1f}h — current data coverage limit."
            return res
            
        except (Exception, SystemExit) as exc:
            logger.warning(f"OpenDrift forecast aborted/failed: {exc}. Using synthetic fallback.")
            return self._synthetic_drift(lon, lat, time_satellite, duration_hours, n_particles, backward=False)

    # ------------------------------------------------------------------
    # Synthetic Lagrangian fallback
    # ------------------------------------------------------------------

    def _synthetic_drift(
        self,
        lon: float,
        lat: float,
        time_start: str,
        duration_hours: int,
        n_particles: int,
        backward: bool = False,
        direction_degrees: float = 225.0,
        current_speed_ms: float = 0.3,
        wind_speed_ms: float = 5.0,
        wind_direction_degrees: float = 225.0,
    ) -> dict:
        """
        Pure-Python synthetic Lagrangian particle drift simulation.

        Does NOT require OpenDrift or any external ocean data.

        Physics:
            At each hourly step:
              u_total = u_current + STOKES_FACTOR · u_wind
              v_total = v_current + STOKES_FACTOR · v_wind

            Convert m/s → degrees:
              Δlon = (u_total · Δt) / (SEC_PER_DEG_LAT · cos(lat))
              Δlat = (v_total · Δt) / SEC_PER_DEG_LAT

            Add turbulent diffusion:
              σ_diff = 0.1 km / √h ≈ 0.001° per hour

        Parameters
        ----------
        lon, lat : float — seed location
        time_start : str — ISO 8601
        duration_hours : int
        n_particles : int
        backward : bool — if True, reverses velocity direction
        direction_degrees : float — ocean current direction (oceanographic)
        current_speed_ms : float
        wind_speed_ms : float
        wind_direction_degrees : float

        Returns
        -------
        dict — GeoJSON FeatureCollection
        """
        logger.info(
            f"Synthetic drift simulation: "
            f"{'backward' if backward else 'forward'} "
            f"{duration_hours}h, {n_particles} particles"
        )

        # ── Try to read forcing from NetCDF if available ───────────────
        try:
            cur_u, cur_v, wind_u, wind_v = self._read_forcing_at(lon, lat)
        except Exception:
            # Build from config defaults
            cur_dir_rad = math.radians(direction_degrees)
            cur_u = current_speed_ms * math.sin(cur_dir_rad)
            cur_v = current_speed_ms * math.cos(cur_dir_rad)

            wind_dir_rad = math.radians(wind_direction_degrees)
            wind_u = -wind_speed_ms * math.sin(wind_dir_rad)  # FROM
            wind_v = -wind_speed_ms * math.cos(wind_dir_rad)

        # Total effective velocity (m/s)
        u_eff = cur_u + STOKES_FACTOR * wind_u
        v_eff = cur_v + STOKES_FACTOR * wind_v

        if backward:
            u_eff = -u_eff
            v_eff = -v_eff

        # ── Simulate particles ─────────────────────────────────────────
        rng = np.random.default_rng(seed=0)
        dt_seconds = 3600.0  # 1-hour time steps

        # Convert m/s → degrees per second
        cos_lat = math.cos(math.radians(lat))
        u_deg_per_s = u_eff / (SEC_PER_DEG_LAT * max(cos_lat, 0.01))
        v_deg_per_s = v_eff / SEC_PER_DEG_LAT

        # Turbulent diffusion standard deviation (degrees per step)
        sigma_diff = 0.001  # ~110 m per hour

        # Seed particles around the centroid (small Gaussian spread)
        lons_p = lon + rng.normal(0, 0.01, n_particles)
        lats_p = lat + rng.normal(0, 0.01, n_particles)

        # Time axis
        t0 = datetime.fromisoformat(time_start.replace("Z", "+00:00"))
        sign = -1 if backward else 1
        times: list[datetime] = [
            t0 + sign * timedelta(hours=h)
            for h in range(duration_hours + 1)
        ]

        # Store trajectories: shape (n_particles, n_times)
        all_lons = np.zeros((n_particles, duration_hours + 1))
        all_lats = np.zeros((n_particles, duration_hours + 1))
        all_lons[:, 0] = lons_p
        all_lats[:, 0] = lats_p

        for t_idx in range(1, duration_hours + 1):
            noise_lon = rng.normal(0, sigma_diff, n_particles)
            noise_lat = rng.normal(0, sigma_diff, n_particles)
            lons_p = lons_p + u_deg_per_s * dt_seconds + noise_lon
            lats_p = lats_p + v_deg_per_s * dt_seconds + noise_lat
            all_lons[:, t_idx] = lons_p
            all_lats[:, t_idx] = lats_p

        result = self._trajectories_to_geojson(all_lons, all_lats, times)
        result["_used_method"] = "synthetic_lagrangian"
        return result

    # ------------------------------------------------------------------
    # Forcing reader helper
    # ------------------------------------------------------------------

    def _read_forcing_at(
        self, lon: float, lat: float
    ) -> tuple[float, float, float, float]:
        """
        Extract average u/v current and wind at a point from the NetCDF files.

        Returns
        -------
        tuple (cur_u, cur_v, wind_u, wind_v) all in m/s
        """
        import netCDF4 as nc

        cur_u = cur_v = wind_u = wind_v = 0.0

        with nc.Dataset(self.ocean_data_path) as ds:
            if "uo" in ds.variables and "vo" in ds.variables:
                cur_u = float(np.nanmean(ds["uo"][:]))
                cur_v = float(np.nanmean(ds["vo"][:]))
            if "u10" in ds.variables and "v10" in ds.variables:
                wind_u = float(np.nanmean(ds["u10"][:]))
                wind_v = float(np.nanmean(ds["v10"][:]))

        # Try wind-only file if different
        if self.wind_data_path != self.ocean_data_path:
            with nc.Dataset(self.wind_data_path) as ds:
                if "u10" in ds.variables and "v10" in ds.variables:
                    wind_u = float(np.nanmean(ds["u10"][:]))
                    wind_v = float(np.nanmean(ds["v10"][:]))

        return cur_u, cur_v, wind_u, wind_v

    # ------------------------------------------------------------------
    # GeoJSON serialization
    # ------------------------------------------------------------------

    def _trajectories_to_geojson(
        self,
        lons: np.ndarray,
        lats: np.ndarray,
        times: list,
    ) -> dict:
        """
        Convert particle trajectory arrays to a GeoJSON FeatureCollection.

        Each particle becomes one GeoJSON LineString Feature.

        Parameters
        ----------
        lons : np.ndarray, shape (n_particles, n_times) — longitudes
        lats : np.ndarray, shape (n_particles, n_times) — latitudes
        times : list[datetime]

        Returns
        -------
        dict — GeoJSON FeatureCollection
        """
        time_strings = [
            t.isoformat() if isinstance(t, datetime) else str(t) for t in times
        ]

        features = []
        for i in range(lons.shape[0]):
            # Filter out NaN coordinates (stranded / deactivated particles)
            coordinates = []
            for t in range(lons.shape[1]):
                lon_val = float(lons[i, t])
                lat_val = float(lats[i, t])
                if math.isnan(lon_val) or math.isnan(lat_val):
                    break  # particle was deactivated at this timestep
                coordinates.append([round(lon_val, 6), round(lat_val, 6)])

            if len(coordinates) < 2:
                continue  # skip particles with too few valid points

            feature = {
                "type": "Feature",
                "geometry": {
                    "type": "LineString",
                    "coordinates": coordinates,
                },
                "properties": {
                    "particle_id": i,
                    "times": time_strings[:len(coordinates)],
                    "start_lon": coordinates[0][0],
                    "start_lat": coordinates[0][1],
                    "end_lon": coordinates[-1][0],
                    "end_lat": coordinates[-1][1],
                },
            }
            features.append(feature)

        return {
            "type": "FeatureCollection",
            "features": features,
            "metadata": {
                "n_particles": len(features),
                "n_timesteps": int(lons.shape[1]),
                "time_start": time_strings[0] if time_strings else "",
                "time_end": time_strings[-1] if time_strings else "",
            },
        }


# ---------------------------------------------------------------------------
# CLI self-test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import json
    from .data_fetcher import OceanDataFetcher

    # Generate synthetic forcing
    fetcher = OceanDataFetcher("data/raw")
    forcing_path = fetcher.get_synthetic_forcing(
        lon=68.5, lat=20.3,
        time_start="2024-06-13T08:30:00",
        time_end="2024-06-16T08:30:00",
        kind="combined",
    )

    engine = DriftEngine(ocean_data_path=forcing_path, wind_data_path=forcing_path)

    hindcast = engine.run_hindcast(
        lon=68.5, lat=20.3,
        time_satellite="2024-06-15T08:30:00Z",
        duration_hours=48,
        n_particles=50,
    )
    logger.info(f"Hindcast features: {len(hindcast['features'])}")

    forecast = engine.run_forecast(
        lon=68.5, lat=20.3,
        time_satellite="2024-06-15T08:30:00Z",
        duration_hours=72,
        n_particles=50,
    )
    logger.info(f"Forecast features: {len(forecast['features'])}")
