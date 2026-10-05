/**
 * Core TypeScript interfaces for the OilGuard maritime intelligence system.
 * Types mirror the FastAPI/Pydantic backend models.
 */

export interface SpillRecord {
  id: string;
  name: string;
  centroid_lon: number;
  centroid_lat: number;
  area_km2: number;
  perimeter_km?: number;
  acquisition_time?: string;
  detection_time?: string;
  status: string;
  age_estimate?: AgeEstimate;
  spill_polygon?: GeoJSONFeature | null;
}

export interface AgeEstimate {
  age_hours_min: number;
  age_hours_max: number;
  age_hours_mean: number;
  method: string;
  confidence?: number;
}

export interface DriftResult {
  spill_id: string;
  origin_lon: number;
  origin_lat: number;
  origin_time: string;
  uncertainty_km: number;
  confidence: number;
  hindcast_geojson: GeoJSONFeatureCollection | string;
  forecast_geojson: GeoJSONFeatureCollection | string;
  warnings?: string[];
}

export interface ScoreBreakdown {
  proximity: number;
  trajectory: number;
  darkness: number;
  vessel_type: number;
  speed_anomaly: number;
  // legacy keys — handled gracefully
  temporal?: number;
  ais_gap?: number;
  course_alignment?: number;
  historical?: number;
}

export interface AISGap {
  start?: string;
  gap_start?: string;
  end?: string;
  gap_end?: string;
  duration_hours?: number;
  duration_minutes?: number;
  lon_before?: number;
  lat_before?: number;
  lon_after?: number;
  lat_after?: number;
}

export interface VesselInfo {
  rank: number;
  mmsi: string;
  vessel_name: string;
  vessel_type: string;
  flag: string;
  total_score: number;
  score_breakdown: ScoreBreakdown;
  proximity_km: number;
  ais_gaps: AISGap[];
  track_geojson?: GeoJSONFeature | GeoJSONGeometry | string | null;
  imo?: string;
  length_m?: number;
}

export interface PipelineStatus {
  stage: string;
  percent: number;
  message: string;
  spill_id?: string;
  error?: string;
}

/** Minimal GeoJSON types */
export interface GeoJSONGeometry {
  type: 'Point' | 'LineString' | 'Polygon' | 'MultiPolygon' | 'MultiLineString';
  coordinates: unknown;
}

export interface GeoJSONFeature {
  type: 'Feature';
  geometry: GeoJSONGeometry;
  properties?: Record<string, unknown>;
}

export interface GeoJSONFeatureCollection {
  type: 'FeatureCollection';
  features: GeoJSONFeature[];
}

export interface DemoResponse {
  message: string;
  spill_id: string;
  spill: SpillRecord;
  drift: DriftResult;
  vessels: VesselInfo[];
  scenario_name: string;
  data_source?: string;
  ais_source?: string;
  sar_source?: string;
  advection_model?: string;
  ais_status?: string;
}

/** Layer visibility toggles */
export interface LayerVisibility {
  spillPolygon: boolean;
  hindcast: boolean;
  forecast: boolean;
  vesselTracks: boolean;
  uncertaintyCircle: boolean;
}

export type LoadingState = 'idle' | 'loading' | 'success' | 'error';

/** Right panel active tab */
export type RightPanelTab = 'vessels' | 'spill' | 'pipeline';
