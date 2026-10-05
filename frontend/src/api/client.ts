import axios from 'axios'
import type {
  SpillRecord, DriftResult, VesselInfo, DemoResponse,
  GeoJSONFeatureCollection,
} from '../types'

const api = axios.create({
  baseURL: '',
  timeout: 120000,
})

// ── Spills ─────────────────────────────────────────────────────────────────

export async function listSpills(): Promise<SpillRecord[]> {
  const { data } = await api.get('/api/spills')
  // Map backend SpillSummary → frontend SpillRecord
  return (data as any[]).map((s: any) => ({
    id:               s.id,
    name:             s.name || `Spill ${s.id}`,
    centroid_lon:     s.origin_lon,
    centroid_lat:     s.origin_lat,
    area_km2:         s.area_km2,
    acquisition_time: s.detection_time,
    detection_time:   s.detection_time,
    status:           'complete',
    confidence:       s.confidence,
  }))
}

export async function getSpillDetail(spillId: string): Promise<any> {
  const { data } = await api.get(`/api/spills/${spillId}`)
  return data
}

// ── Vessels ────────────────────────────────────────────────────────────────

export async function getVessels(spillId: string): Promise<VesselInfo[]> {
  const { data } = await api.get(`/api/vessels/${spillId}`)
  return (data as any[]).map((v: any) => ({
    rank:            v.rank,
    mmsi:            String(v.mmsi),
    vessel_name:     v.vessel_name,
    vessel_type:     v.vessel_type,
    flag:            v.flag || 'IN',
    total_score:     v.total_score,
    score_breakdown: v.score_breakdown,
    proximity_km:    v.closest_approach_km,
    ais_gaps:        v.ais_gaps || [],
    track_geojson:   null, // Loaded lazily via /api/ais/{mmsi}
  }))
}

// ── AIS ────────────────────────────────────────────────────────────────────

export async function getAISTrack(mmsi: string): Promise<any> {
  const { data } = await api.get(`/api/ais/${mmsi}`)
  return data
}

export async function getInterpolatedPosition(
  mmsi: string, timestamp: string
): Promise<{ lat: number; lon: number; heading: number; sog: number }> {
  const { data } = await api.get(`/api/ais/${mmsi}/at`, {
    params: { t: timestamp },
  })
  return data
}

// ── Ocean ──────────────────────────────────────────────────────────────────

export async function getOceanGrid(timestamp?: string): Promise<any[]> {
  const { data } = await api.get('/api/ocean/grid', {
    params: timestamp ? { t: timestamp } : {},
  })
  return data
}

// ── Detection (upload) ─────────────────────────────────────────────────────

export async function detectSpill(formData: FormData): Promise<SpillRecord> {
  const { data } = await api.post('/api/demo/detect', formData, {
    headers: { 'Content-Type': 'multipart/form-data' },
    timeout: 60000,
  })
  return { ...data, id: data.spill_id }
}

// ── Drift (trigger) ────────────────────────────────────────────────────────

export async function runDrift(spillId: string): Promise<DriftResult> {
  const { data } = await api.post(`/api/drift/${spillId}`)
  return data
}

// ── Demo ───────────────────────────────────────────────────────────────────

export async function generateSyntheticDemo(): Promise<DemoResponse> {
  const { data } = await api.get('/api/demo/generate')
  return data as DemoResponse
}

// ── Incidents (Multi-Mission) ──────────────────────────────────────────────

export async function listIncidents(): Promise<any[]> {
  const { data } = await api.get('/api/incidents')
  return data
}

export async function detectIncident(incidentId: string, spillId?: string): Promise<DemoResponse> {
  const payload = spillId ? { spill_id: spillId } : {}
  const { data } = await api.post(`/api/incidents/${incidentId}/detect`, payload, {
    timeout: 120000,  // OpenDrift physics runs take 30-40s
  })
  return data as DemoResponse
}

// ── Health ──────────────────────────────────────────────────────────────────

export async function health(): Promise<{ status: string }> {
  const { data } = await api.get('/health')
  return data
}

// Legacy object-style export for backward compat
export const apiClient = {
  detectSpill, runDrift, getVessels, listSpills,
  getAISTrack, generateSyntheticDemo, detectIncident, listIncidents, health,
}
