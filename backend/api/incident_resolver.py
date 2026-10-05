"""
backend/api/incident_resolver.py
=================================
Sole filesystem authority for resolving incident data paths.
No other module should construct paths to incident data directly.
"""

from pathlib import Path
from typing import Optional
import yaml
from fastapi import HTTPException
from loguru import logger

# Incidents live under data/incidents/ at the project root
PROJECT_ROOT = Path(__file__).resolve().parents[2]
INCIDENTS_DIR = PROJECT_ROOT / "data" / "incidents"


class IncidentResolver:
    """
    Resolves and validates file paths for a specific incident.
    
    Usage:
        resolver = IncidentResolver("ennore_2017")
        resolver.validate()  # raises 404 if missing
        config = resolver.get_config()
        nc_path = resolver.get_ocean_path()
    """

    def __init__(self, incident_id: str):
        self.incident_id = incident_id
        self.base_path = INCIDENTS_DIR / incident_id

    def validate(self) -> None:
        """Raises HTTPException 404 if the incident directory or config is missing."""
        if not self.base_path.is_dir():
            raise HTTPException(
                status_code=404,
                detail=f"Incident '{self.incident_id}' not found. "
                       f"Available: {[d.name for d in INCIDENTS_DIR.iterdir() if d.is_dir()]}"
            )
        config_path = self.base_path / "incident.yaml"
        if not config_path.is_file():
            raise HTTPException(
                status_code=404,
                detail=f"Incident '{self.incident_id}' is missing incident.yaml config."
            )

    def get_config(self) -> dict:
        """Load and return the incident.yaml configuration."""
        config_path = self.base_path / "incident.yaml"
        with open(config_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
        logger.info(f"Loaded incident config: {self.incident_id}")
        return config

    def get_sar_path(self) -> Optional[Path]:
        """Returns path to SAR image, or None if not present."""
        candidates = list(self.base_path.glob("sar_image.*")) + list(self.base_path.glob("*.tif"))
        if candidates:
            return candidates[0]
        return None

    def get_ocean_path(self) -> Optional[Path]:
        """Returns path to ocean currents NetCDF, or None."""
        nc_path = self.base_path / "ocean_currents.nc"
        if nc_path.is_file():
            return nc_path
        return None

    def get_ais_path(self) -> Optional[Path]:
        """Returns path to AIS traffic CSV, or None."""
        csv_path = self.base_path / "ais_traffic.csv"
        if csv_path.is_file():
            return csv_path
        return None

    @staticmethod
    def list_incidents() -> list[dict]:
        """
        Scan the incidents/ directory and return metadata for all available incidents.
        Returns a list of dicts with id, name, date, location, data_source.
        """
        incidents = []
        if not INCIDENTS_DIR.is_dir():
            return incidents

        for d in sorted(INCIDENTS_DIR.iterdir()):
            if not d.is_dir():
                continue
            config_path = d / "incident.yaml"
            if not config_path.is_file():
                continue
            try:
                with open(config_path, "r", encoding="utf-8") as f:
                    cfg = yaml.safe_load(f)
                incident_info = cfg.get("incident", {})
                spill_info = cfg.get("spill", {})
                incidents.append({
                    "id": incident_info.get("id", d.name),
                    "name": incident_info.get("name", d.name),
                    "description": incident_info.get("description", ""),
                    "date": incident_info.get("date", ""),
                    "location": [
                        spill_info.get("centroid_lon", 0),
                        spill_info.get("centroid_lat", 0),
                    ],
                    "data_source": incident_info.get("data_source", "unknown"),
                    "has_sar": (d / "sar_image.tif").is_file() or bool(list(d.glob("*.tif"))),
                    "has_ocean": (d / "ocean_currents.nc").is_file(),
                    "has_ais": (d / "ais_traffic.csv").is_file(),
                })
            except Exception as e:
                logger.warning(f"Failed to load incident config from {d}: {e}")
                continue

        return incidents
