"""
backend/drift/data_fetcher.py
==============================
Ocean & Atmospheric Forcing Data Fetcher.

Supports:
  1. CMEMS (Copernicus Marine Environment Monitoring Service)
     — Ocean surface currents (u, v components)
  2. ERA5 via CDS API
     — 10m wind speed (u10, v10 components)
  3. Synthetic fallback
     — Deterministic + noise synthetic forcing fields for offline testing

All methods return the path to a NetCDF file suitable for use by OpenDrift.
"""

from __future__ import annotations

import os
import warnings
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

import numpy as np
from loguru import logger

try:
    import netCDF4 as nc
    NC4_AVAILABLE = True
except ImportError:
    NC4_AVAILABLE = False
    logger.warning("netCDF4 not installed. Synthetic NetCDF output unavailable.")

try:
    import xarray as xr
    XR_AVAILABLE = True
except ImportError:
    XR_AVAILABLE = False


class OceanDataFetcher:
    """
    Fetches ocean current and wind forcing data for drift simulation.

    Parameters
    ----------
    cache_dir : str
        Directory where downloaded / generated NetCDF files are cached.
        Created automatically if it doesn't exist.
    """

    def __init__(self, cache_dir: str = "data/raw") -> None:
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # CMEMS Ocean Currents
    # ------------------------------------------------------------------

    def fetch_cmems(
        self,
        lon_min: float,
        lon_max: float,
        lat_min: float,
        lat_max: float,
        time_start: str,
        time_end: str,
        output_path: Optional[str] = None,
    ) -> str:
        """
        Download ocean surface current data from CMEMS.

        Uses the ``copernicusmarine`` Python client if installed; falls back to
        the OPeNDAP URL approach otherwise.

        Credentials are read from environment variables:
          - CMEMS_USERNAME
          - CMEMS_PASSWORD

        Dataset used: ``cmems_mod_glo_phy_anfc_merged-uv_PT1H-i``
        (Global Ocean Physics Analysis and Forecast, hourly, 1/12°)

        Parameters
        ----------
        lon_min, lon_max, lat_min, lat_max : float
            Bounding box in WGS84 degrees.
        time_start, time_end : str
            ISO 8601 datetime strings (e.g. '2024-06-14T00:00:00').
        output_path : str, optional
            Full path for the output NetCDF.  Auto-generated if None.

        Returns
        -------
        str
            Path to the downloaded NetCDF file.
        """
        if output_path is None:
            ts = time_start.replace(":", "").replace("-", "")[:12]
            output_path = str(self.cache_dir / f"cmems_currents_{ts}.nc")

        username = os.getenv("CMEMS_USERNAME", "")
        password = os.getenv("CMEMS_PASSWORD", "")

        if not username or not password:
            logger.warning(
                "CMEMS credentials not set (CMEMS_USERNAME / CMEMS_PASSWORD). "
                "Generating synthetic ocean data instead."
            )
            return self.get_synthetic_forcing(
                lon=(lon_min + lon_max) / 2,
                lat=(lat_min + lat_max) / 2,
                time_start=time_start,
                time_end=time_end,
                output_path=output_path,
                kind="ocean",
            )

        try:
            import copernicusmarine  # type: ignore

            logger.info(f"Downloading CMEMS data: [{lon_min},{lat_min}] → [{lon_max},{lat_max}]")
            copernicusmarine.subset(
                dataset_id="cmems_mod_glo_phy_anfc_merged-uv_PT1H-i",
                variables=["uo", "vo"],
                minimum_longitude=lon_min,
                maximum_longitude=lon_max,
                minimum_latitude=lat_min,
                maximum_latitude=lat_max,
                start_datetime=time_start,
                end_datetime=time_end,
                output_filename=output_path,
                username=username,
                password=password,
                force_download=True,
            )
            logger.info(f"CMEMS data saved to {output_path}")
            return output_path

        except Exception as exc:
            logger.error(f"CMEMS download failed: {exc}. Using synthetic data.")
            return self.get_synthetic_forcing(
                lon=(lon_min + lon_max) / 2,
                lat=(lat_min + lat_max) / 2,
                time_start=time_start,
                time_end=time_end,
                output_path=output_path,
                kind="ocean",
            )

    # ------------------------------------------------------------------
    # ERA5 Wind
    # ------------------------------------------------------------------

    def fetch_era5_wind(
        self,
        lon_min: float,
        lon_max: float,
        lat_min: float,
        lat_max: float,
        time_start: str,
        time_end: str,
        output_path: Optional[str] = None,
    ) -> str:
        """
        Download ERA5 10-metre wind field data via the CDS API.

        Requires the ``cdsapi`` package and a valid ``~/.cdsapirc`` file with
        CDS credentials.

        Parameters
        ----------
        lon_min, lon_max, lat_min, lat_max : float
        time_start, time_end : str  — ISO 8601
        output_path : str, optional

        Returns
        -------
        str — path to NetCDF file.
        """
        if output_path is None:
            ts = time_start.replace(":", "").replace("-", "")[:12]
            output_path = str(self.cache_dir / f"era5_wind_{ts}.nc")

        try:
            import cdsapi  # type: ignore

            t0 = datetime.fromisoformat(time_start.replace("Z", "+00:00"))
            t1 = datetime.fromisoformat(time_end.replace("Z", "+00:00"))

            # Build hour list
            hours = []
            cur = t0
            while cur <= t1:
                hours.append(cur.strftime("%H:%M"))
                cur += timedelta(hours=1)
            hours = list(set(hours))

            dates = [
                (t0 + timedelta(days=d)).strftime("%Y-%m-%d")
                for d in range((t1.date() - t0.date()).days + 1)
            ]

            logger.info(f"Downloading ERA5 wind data for {dates}")
            c = cdsapi.Client()
            c.retrieve(
                "reanalysis-era5-single-levels",
                {
                    "product_type": "reanalysis",
                    "variable": ["10m_u_component_of_wind", "10m_v_component_of_wind"],
                    "year": list({d[:4] for d in dates}),
                    "month": list({d[5:7] for d in dates}),
                    "day": list({d[8:10] for d in dates}),
                    "time": hours,
                    "area": [lat_max, lon_min, lat_min, lon_max],
                    "format": "netcdf",
                },
                output_path,
            )
            logger.info(f"ERA5 wind saved to {output_path}")
            return output_path

        except Exception as exc:
            logger.error(f"ERA5 download failed: {exc}. Using synthetic wind data.")
            return self.get_synthetic_forcing(
                lon=(lon_min + lon_max) / 2,
                lat=(lat_min + lat_max) / 2,
                time_start=time_start,
                time_end=time_end,
                output_path=output_path,
                kind="wind",
            )

    # ------------------------------------------------------------------
    # Synthetic Forcing (fallback)
    # ------------------------------------------------------------------

    def get_synthetic_forcing(
        self,
        lon: float,
        lat: float,
        time_start: str,
        time_end: str,
        output_path: Optional[str] = None,
        kind: str = "combined",
        wind_speed_ms: float = 5.0,
        wind_dir_deg: float = 225.0,
        current_speed_ms: float = 0.3,
        current_dir_deg: float = 180.0,
    ) -> str:
        """
        Generate a simple synthetic forcing NetCDF for offline testing.

        Creates a 5×5 degree regular grid around (lon, lat) with uniform
        wind and current fields + small random perturbations.

        Parameters
        ----------
        lon, lat : float
            Centre point of the synthetic domain.
        time_start, time_end : str — ISO 8601
        output_path : str, optional
        kind : str — 'ocean', 'wind', or 'combined'
        wind_speed_ms : float — wind speed magnitude (m/s)
        wind_dir_deg : float — meteorological wind direction (degrees from N)
        current_speed_ms : float — current speed (m/s)
        current_dir_deg : float — current direction (degrees from N)

        Returns
        -------
        str — path to the generated NetCDF file.
        """
        if output_path is None:
            ts = time_start.replace(":", "").replace("-", "")[:12]
            output_path = str(self.cache_dir / f"synthetic_forcing_{kind}_{ts}.nc")

        if not NC4_AVAILABLE:
            logger.warning("netCDF4 not available; returning dummy path for synthetic forcing.")
            # Write a minimal JSON-like placeholder so the engine can detect it
            with open(output_path.replace(".nc", ".txt"), "w") as f:
                f.write(f"synthetic:{kind}:{wind_speed_ms}:{wind_dir_deg}:{current_speed_ms}:{current_dir_deg}")
            return output_path

        # ── Build grid ──────────────────────────────────────────────────
        # 5×5 degree domain, 0.1° resolution
        lons = np.arange(lon - 2.5, lon + 2.5, 0.1, dtype=np.float64)
        lats = np.arange(lat - 2.5, lat + 2.5, 0.1, dtype=np.float64)
        NX, NY = len(lons), len(lats)

        # ── Build time axis ─────────────────────────────────────────────
        t0 = datetime.fromisoformat(time_start.replace("Z", "+00:00"))
        t1 = datetime.fromisoformat(time_end.replace("Z", "+00:00"))
        hours_total = max(1, int((t1 - t0).total_seconds() / 3600))
        times_h = np.arange(0, hours_total + 1, dtype=np.float64)  # hours since t0
        NT = len(times_h)

        # ── Convert directions to u/v components ────────────────────────
        # Oceanographic convention: direction is "going TO" (current)
        # Meteorological convention: direction is "coming FROM" (wind)

        def _dir_to_uv(speed: float, dir_deg: float, met: bool = False) -> tuple[float, float]:
            """Convert speed + direction to (u, v) components."""
            dir_rad = np.radians(dir_deg)
            if met:
                # Wind comes FROM dir_deg → goes toward (dir_deg + 180)°
                u = -speed * np.sin(dir_rad)
                v = -speed * np.cos(dir_rad)
            else:
                # Current goes TOWARD dir_deg
                u = speed * np.sin(dir_rad)
                v = speed * np.cos(dir_rad)
            return u, v

        cur_u, cur_v = _dir_to_uv(current_speed_ms, current_dir_deg, met=False)
        wind_u, wind_v = _dir_to_uv(wind_speed_ms, wind_dir_deg, met=True)

        # ── Write NetCDF ─────────────────────────────────────────────────
        rng = np.random.default_rng(seed=42)

        with nc.Dataset(output_path, "w", format="NETCDF4") as ds:
            # Dimensions
            ds.createDimension("time", NT)
            ds.createDimension("lat", NY)
            ds.createDimension("lon", NX)

            # Coordinate variables
            time_var = ds.createVariable("time", "f8", ("time",))
            time_var.units = f"hours since {t0.strftime('%Y-%m-%d %H:%M:%S')}"
            time_var.calendar = "standard"
            time_var[:] = times_h

            lat_var = ds.createVariable("lat", "f4", ("lat",))
            lat_var.units = "degrees_north"
            lat_var[:] = lats.astype(np.float32)

            lon_var = ds.createVariable("lon", "f4", ("lon",))
            lon_var.units = "degrees_east"
            lon_var[:] = lons.astype(np.float32)

            if kind in ("ocean", "combined"):
                # Ocean current U component (m/s)
                uo = ds.createVariable("uo", "f4", ("time", "lat", "lon"), fill_value=9.969e+36)
                uo.units = "m s-1"
                uo.long_name = "Eastward sea water velocity"
                noise_u = rng.normal(0, 0.02, (NT, NY, NX)).astype(np.float32)
                uo[:] = np.full((NT, NY, NX), cur_u, dtype=np.float32) + noise_u

                vo = ds.createVariable("vo", "f4", ("time", "lat", "lon"), fill_value=9.969e+36)
                vo.units = "m s-1"
                vo.long_name = "Northward sea water velocity"
                noise_v = rng.normal(0, 0.02, (NT, NY, NX)).astype(np.float32)
                vo[:] = np.full((NT, NY, NX), cur_v, dtype=np.float32) + noise_v

            if kind in ("wind", "combined"):
                # 10m wind U component (m/s)
                u10 = ds.createVariable("u10", "f4", ("time", "lat", "lon"), fill_value=9.969e+36)
                u10.units = "m s-1"
                u10.long_name = "10 metre U wind component"
                noise_wu = rng.normal(0, 0.3, (NT, NY, NX)).astype(np.float32)
                u10[:] = np.full((NT, NY, NX), wind_u, dtype=np.float32) + noise_wu

                v10 = ds.createVariable("v10", "f4", ("time", "lat", "lon"), fill_value=9.969e+36)
                v10.units = "m s-1"
                v10.long_name = "10 metre V wind component"
                noise_wv = rng.normal(0, 0.3, (NT, NY, NX)).astype(np.float32)
                v10[:] = np.full((NT, NY, NX), wind_v, dtype=np.float32) + noise_wv

            # Global attributes
            ds.title = "Synthetic ocean forcing for oil spill drift simulation"
            ds.institution = "Oil Spill Detection System (NTRO #26143)"
            ds.source = "Synthetic"
            ds.Conventions = "CF-1.8"
            ds.history = f"Generated {datetime.utcnow().isoformat()}Z"

        logger.info(f"Synthetic {kind} forcing saved to {output_path} ({NT} time steps, {NY}×{NX} grid)")
        return output_path


# ---------------------------------------------------------------------------
# CLI self-test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    fetcher = OceanDataFetcher(cache_dir="data/raw")
    path = fetcher.get_synthetic_forcing(
        lon=68.5,
        lat=20.3,
        time_start="2024-06-13T08:30:00",
        time_end="2024-06-16T08:30:00",
        kind="combined",
    )
    logger.info(f"Synthetic forcing generated: {path}")
