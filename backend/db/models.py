"""
SQLAlchemy 2.0 ORM models for oil spill system.
"""
from datetime import datetime
from sqlalchemy import String, Float, Text, DateTime, ForeignKey
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship
import uuid

class Base(DeclarativeBase):
    pass

class SpillRecord(Base):
    __tablename__ = 'spill_records'
    id: Mapped[str] = mapped_column(String, primary_key=True, default=lambda: str(uuid.uuid4())[:8])
    name: Mapped[str] = mapped_column(String, nullable=False)
    sar_path: Mapped[str] = mapped_column(String, nullable=True)
    acquisition_time: Mapped[str] = mapped_column(String, nullable=True)
    detection_time: Mapped[str] = mapped_column(String, nullable=True)
    centroid_lon: Mapped[float] = mapped_column(Float, nullable=True)
    centroid_lat: Mapped[float] = mapped_column(Float, nullable=True)
    area_km2: Mapped[float] = mapped_column(Float, nullable=True)
    perimeter_km: Mapped[float] = mapped_column(Float, nullable=True)
    age_hours_mean: Mapped[float] = mapped_column(Float, nullable=True)
    spill_polygon_json: Mapped[str] = mapped_column(Text, nullable=True)  # GeoJSON string
    status: Mapped[str] = mapped_column(String, default='pending')  # pending/analysed/error
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    drift_result: Mapped['DriftResult'] = relationship(back_populates='spill', uselist=False)
    vessel_results: Mapped[list['VesselResult']] = relationship(back_populates='spill')

class DriftResult(Base):
    __tablename__ = 'drift_results'
    id: Mapped[str] = mapped_column(String, primary_key=True, default=lambda: str(uuid.uuid4())[:8])
    spill_id: Mapped[str] = mapped_column(String, ForeignKey('spill_records.id'), nullable=False)
    hindcast_geojson: Mapped[str] = mapped_column(Text, nullable=True)
    forecast_geojson: Mapped[str] = mapped_column(Text, nullable=True)
    origin_lon: Mapped[float] = mapped_column(Float, nullable=True)
    origin_lat: Mapped[float] = mapped_column(Float, nullable=True)
    origin_time: Mapped[str] = mapped_column(String, nullable=True)
    uncertainty_km: Mapped[float] = mapped_column(Float, nullable=True)
    confidence: Mapped[float] = mapped_column(Float, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    spill: Mapped['SpillRecord'] = relationship(back_populates='drift_result')

class VesselResult(Base):
    __tablename__ = 'vessel_results'
    id: Mapped[str] = mapped_column(String, primary_key=True, default=lambda: str(uuid.uuid4())[:8])
    spill_id: Mapped[str] = mapped_column(String, ForeignKey('spill_records.id'), nullable=False)
    mmsi: Mapped[str] = mapped_column(String, nullable=False)
    vessel_name: Mapped[str] = mapped_column(String, nullable=True)
    vessel_type: Mapped[str] = mapped_column(String, nullable=True)
    flag: Mapped[str] = mapped_column(String, nullable=True)
    total_score: Mapped[float] = mapped_column(Float, nullable=True)
    score_breakdown_json: Mapped[str] = mapped_column(Text, nullable=True)
    track_geojson: Mapped[str] = mapped_column(Text, nullable=True)
    rank: Mapped[int] = mapped_column(nullable=True)
    proximity_km: Mapped[float] = mapped_column(Float, nullable=True)
    ais_gaps_json: Mapped[str] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    spill: Mapped['SpillRecord'] = relationship(back_populates='vessel_results')
