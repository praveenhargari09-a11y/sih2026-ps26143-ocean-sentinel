/**
 * SpillLayer — spill polygon with pulsing outline, origin ping marker,
 * and optional uncertainty circle. Uses react-leaflet's Marker properly.
 */
import React, { useMemo } from 'react'
import { GeoJSON, Circle, Marker } from 'react-leaflet'
import L from 'leaflet'
import type { SpillRecord, DriftResult } from '../types'

interface Props {
  spill: SpillRecord | null
  drift: DriftResult | null
  showUncertainty?: boolean
}

function parseGeoJSON(raw: unknown): any {
  if (!raw) return null
  if (typeof raw === 'string') { try { return JSON.parse(raw) } catch { return null } }
  return raw
}

const originIcon = L.divIcon({
  className: '',
  iconSize:   [24, 24],
  iconAnchor: [12, 12],
  html: `<div style="position:relative;width:24px;height:24px;">
    <div style="position:absolute;inset:0;border-radius:50%;background:rgba(248,81,73,0.45);animation:origin-ping 2s ease-out infinite;"></div>
    <div style="position:absolute;inset:6px;border-radius:50%;background:#f85149;border:2px solid #0d1117;"></div>
  </div>`,
})

const centroidIcon = L.divIcon({
  className: '',
  iconSize:   [8, 8],
  iconAnchor: [4, 4],
  html: `<div style="width:8px;height:8px;border-radius:50%;background:#f85149;border:1.5px solid #0d1117;box-shadow:0 0 6px rgba(248,81,73,0.6);"></div>`,
})

export default function SpillLayer({ spill, drift, showUncertainty = true }: Props) {
  const polygon = useMemo(() => {
    if (!spill) return null
    const parsed = parseGeoJSON(spill.spill_polygon)
    if (parsed) return parsed
    // Generate fallback ellipse from centroid + area
    const { centroid_lon: lon, centroid_lat: lat, area_km2 } = spill
    const r = Math.sqrt((area_km2 ?? 10) / Math.PI) / 111
    const pts = Array.from({ length: 32 }, (_, i) => {
      const a = (i / 32) * 2 * Math.PI
      return [lon + r * 1.6 * Math.cos(a), lat + r * Math.sin(a)]
    })
    pts.push(pts[0])
    return { type: 'FeatureCollection', features: [{ type: 'Feature', geometry: { type: 'Polygon', coordinates: [pts] }, properties: {} }] }
  }, [spill])

  if (!spill) return null

  return (
    <>
      {polygon && (
        <GeoJSON
          key={spill.id + '-poly'}
          data={polygon}
          style={{ color: '#f85149', weight: 2, opacity: 1, fillColor: '#f85149', fillOpacity: 0.1, className: 'spill-active-path' }}
        />
      )}

      {drift && showUncertainty && drift.uncertainty_km > 0 && (
        <Circle
          center={[drift.origin_lat, drift.origin_lon]}
          radius={drift.uncertainty_km * 1000}
          pathOptions={{ color: '#8b949e', weight: 1, opacity: 0.45, dashArray: '4 5', fillColor: '#8b949e', fillOpacity: 0.03 }}
        />
      )}

      {drift && <Marker position={[drift.origin_lat, drift.origin_lon]} icon={originIcon} />}

      <Marker position={[spill.centroid_lat, spill.centroid_lon]} icon={centroidIcon} />
    </>
  )
}
