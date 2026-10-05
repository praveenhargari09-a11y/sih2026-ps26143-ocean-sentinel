/**
 * DriftLayer — renders hindcast and forecast trajectories with deliberate
 * visual distinction:
 *   • Hindcast: dashed, muted gray (#4b5563) — "already known, past"
 *   • Forecast: segmented polylines fading amber → transparent — "uncertain future"
 *
 * Both are rendered from GeoJSON FeatureCollections (particle tracks).
 * We extract all LineString coordinates and build an envelope path.
 */
import React, { useMemo } from 'react'
import { Polyline } from 'react-leaflet'
import type { DriftResult, GeoJSONFeatureCollection, GeoJSONFeature } from '../types'

interface Props {
  drift: DriftResult | null
  showHindcast: boolean
  showForecast: boolean
}

/** Parse GeoJSON from string or object */
function parseGeoJSON(raw: unknown): GeoJSONFeatureCollection | null {
  if (!raw) return null
  if (typeof raw === 'string') {
    try { return JSON.parse(raw) } catch { return null }
  }
  return raw as GeoJSONFeatureCollection
}

/** Extract all coordinate arrays from LineString features */
function extractTracks(fc: GeoJSONFeatureCollection): [number, number][][] {
  return fc.features
    .filter(f => f.geometry?.type === 'LineString')
    .map(f => (f.geometry.coordinates as ([number, number] | null)[])
      .filter((c): c is [number, number] => c != null && c[0] != null && c[1] != null && isFinite(c[0]) && isFinite(c[1]))
      .map(([lon, lat]) => [lat, lon] as [number, number])
    )
    .filter(t => t.length >= 2)
}

/** Build an envelope path: average all track positions per time-step */
function buildEnvelope(tracks: [number, number][][]): [number, number][] {
  if (!tracks.length) return []
  const maxLen = Math.max(...tracks.map(t => t.length))
  const result: [number, number][] = []
  for (let i = 0; i < maxLen; i++) {
    const pts = tracks.filter(t => i < t.length).map(t => t[i])
    if (!pts.length) continue
    const lat = pts.reduce((s, p) => s + p[0], 0) / pts.length
    const lon = pts.reduce((s, p) => s + p[1], 0) / pts.length
    result.push([lat, lon])
  }
  return result
}

/** Split path into N segments with linearly decreasing opacity (for forecast fade) */
function buildForecastSegments(
  path: [number, number][],
  nSegments = 10
): { coords: [number, number][]; opacity: number }[] {
  if (path.length < 2) return []
  const segSize = Math.max(2, Math.ceil(path.length / nSegments))
  const segments: { coords: [number, number][]; opacity: number }[] = []
  for (let i = 0; i < nSegments; i++) {
    const start = i * (segSize - 1)
    const end   = Math.min(start + segSize, path.length)
    if (start >= path.length - 1) break
    const slice = path.slice(start, end)
    if (slice.length < 2) continue
    // opacity 0.95 at start, 0.08 at end
    const opacity = 0.95 - (i / (nSegments - 1)) * 0.87
    segments.push({ coords: slice, opacity })
  }
  return segments
}

export default function DriftLayer({ drift, showHindcast, showForecast }: Props) {
  const hindcastTracks = useMemo(() => {
    const fc = parseGeoJSON(drift?.hindcast_geojson)
    if (!fc) return []
    return extractTracks(fc)
  }, [drift?.hindcast_geojson])

  const forecastTracks = useMemo(() => {
    const fc = parseGeoJSON(drift?.forecast_geojson)
    if (!fc) return []
    return extractTracks(fc)
  }, [drift?.forecast_geojson])

  const hindcastPath  = useMemo(() => buildEnvelope(hindcastTracks), [hindcastTracks])
  const forecastPath  = useMemo(() => buildEnvelope(forecastTracks), [forecastTracks])
  const forecastSegs  = useMemo(() => buildForecastSegments(forecastPath, 12), [forecastPath])

  if (!drift) return null

  return (
    <>
      {/* ── Hindcast: dashed gray — the known past ── */}
      {showHindcast && hindcastPath.length >= 2 && (
        <Polyline
          positions={hindcastPath}
          pathOptions={{
            color: '#6b7280',
            weight: 2,
            opacity: 0.7,
            dashArray: '6 5',
            lineCap: 'round',
          }}
        />
      )}

      {/* Hindcast individual particle tracks (very faint, give sense of spread) */}
      {showHindcast && hindcastTracks.slice(0, 40).map((track, i) => (
        <Polyline
          key={`hc-${i}`}
          positions={track}
          pathOptions={{
            color: '#374151',
            weight: 1,
            opacity: 0.25,
            dashArray: '3 6',
          }}
        />
      ))}

      {/* ── Forecast: segmented amber fading to transparent — the uncertain future ── */}
      {showForecast && forecastSegs.map((seg, i) => (
        <Polyline
          key={`fc-seg-${i}`}
          positions={seg.coords}
          pathOptions={{
            color: '#f59e0b',
            weight: i === 0 ? 3 : 2,
            opacity: seg.opacity,
            lineCap: 'round',
            lineJoin: 'round',
          }}
        />
      ))}

      {/* Forecast particle tracks — slightly warm tinted, very faint */}
      {showForecast && forecastTracks.slice(0, 40).map((track, i) => (
        <Polyline
          key={`fc-ptk-${i}`}
          positions={track}
          pathOptions={{
            color: '#92400e',
            weight: 1,
            opacity: 0.15,
          }}
        />
      ))}
    </>
  )
}
