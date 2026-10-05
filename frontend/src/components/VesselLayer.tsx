/**
 * VesselLayer — AIS track polylines (react-leaflet) + position markers (imperative Leaflet).
 * scrubIdx is owned by MapView and passed down so the scrubber overlay stays outside MapContainer.
 */
import React, { useEffect, useRef } from 'react'
import { Polyline, useMap } from 'react-leaflet'
import L from 'leaflet'
import type { VesselInfo, GeoJSONFeature, GeoJSONGeometry } from '../types'

export interface VesselLayerProps {
  vessels: VesselInfo[]
  selectedMMSI: string | null
  onVesselClick: (mmsi: string | null) => void
  scrubIdx?: number
}

/** Parse track_geojson → Leaflet [lat, lon][] */
export function parseTrackCoords(track: VesselInfo['track_geojson']): [number, number][] {
  if (!track) return []
  let geom: GeoJSONGeometry | null = null
  try {
    if (typeof track === 'string') {
      const p = JSON.parse(track)
      geom = p?.geometry ?? p
    } else if ((track as GeoJSONFeature).type === 'Feature') {
      geom = (track as GeoJSONFeature).geometry
    } else {
      geom = track as GeoJSONGeometry
    }
  } catch { return [] }
  if (!geom || geom.type !== 'LineString') return []
  return (geom.coordinates as [number, number][]).map(([lon, lat]) => [lat, lon])
}

function scoreColor(score: number, isSelected: boolean, isTop: boolean): string {
  if (isSelected) return '#22d3ee'
  if (isTop)      return '#fbbf24'
  if (score > 0.7) return '#f85149'
  if (score > 0.4) return '#f59e0b'
  return '#4b5563'
}

function makeMarkerIcon(color: string, isTop: boolean, isSelected: boolean) {
  const size   = isSelected ? 14 : isTop ? 12 : 9
  const shadow = isSelected
    ? 'box-shadow:0 0 0 3px rgba(34,211,238,0.3);'
    : isTop
    ? 'box-shadow:0 0 0 3px rgba(251,191,36,0.3),0 0 10px rgba(251,191,36,0.4);'
    : ''
  return L.divIcon({
    className: '',
    iconSize:   [size, size],
    iconAnchor: [size / 2, size / 2],
    html: `<div style="width:${size}px;height:${size}px;border-radius:50%;border:2px solid ${color};background:#0d1117;cursor:pointer;${shadow}"></div>`,
  })
}

export default function VesselLayer({ vessels, selectedMMSI, onVesselClick, scrubIdx = 0 }: VesselLayerProps) {
  const map        = useMap()
  const markersRef = useRef<Map<string, L.Marker>>(new Map())
  const prevSelRef = useRef<string | null>(null)

  // Pan to selected vessel on first selection (not on every scrub)
  useEffect(() => {
    if (selectedMMSI === prevSelRef.current) return
    prevSelRef.current = selectedMMSI
    const vessel = vessels.find(v => v.mmsi === selectedMMSI)
    if (!vessel) return
    const coords = parseTrackCoords(vessel.track_geojson)
    if (coords.length) map.panTo(coords[coords.length - 1], { animate: true, duration: 0.6 })
  }, [selectedMMSI, vessels, map])

  // Imperative markers — avoids React re-render overhead during scrub
  useEffect(() => {
    markersRef.current.forEach(m => m.remove())
    markersRef.current.clear()

    vessels.forEach(vessel => {
      const coords     = parseTrackCoords(vessel.track_geojson)
      if (!coords.length) return
      const isSelected = vessel.mmsi === selectedMMSI
      const isTop      = vessel.rank === 1
      const color      = scoreColor(vessel.total_score, isSelected, isTop)
      const idx        = isSelected ? Math.min(scrubIdx, coords.length - 1) : coords.length - 1
      const pos        = coords[idx]

      const marker = L.marker(pos, {
        icon: makeMarkerIcon(color, isTop, isSelected),
        zIndexOffset: isTop ? 1000 : isSelected ? 500 : 0,
      })

      marker.on('click', () =>
        onVesselClick(vessel.mmsi === selectedMMSI ? null : vessel.mmsi))

      const scoreBar = `<div style="margin-top:6px;height:3px;background:#21262d;border-radius:2px">
        <div style="width:${(vessel.total_score * 100).toFixed(0)}%;height:100%;background:${color};border-radius:2px"></div>
      </div>`

      marker.bindPopup(
        `<div style="font-family:'Inter',sans-serif;font-size:12px;min-width:160px">
          <div style="color:#8b949e;font-size:10px">MMSI: <span style="font-family:'JetBrains Mono',monospace;color:#22d3ee">${vessel.mmsi}</span></div>
          <div style="color:#e6edf3;font-weight:600;margin:3px 0 1px">${vessel.vessel_name || 'Unknown'}</div>
          <div style="color:#8b949e">${vessel.vessel_type} · ${vessel.flag}</div>
          ${scoreBar}
          <div style="margin-top:3px;font-family:'JetBrains Mono',monospace;font-size:11px;color:${color}">Score ${(vessel.total_score * 100).toFixed(0)}</div>
        </div>`,
        { className: '' }
      )
      
      marker.bindTooltip(
        `<span style="font-family:'Inter',sans-serif;font-weight:600;color:#e6edf3;text-shadow:0 1px 4px rgba(0,0,0,0.8)">${vessel.vessel_name || vessel.mmsi}</span>`,
        { permanent: true, direction: 'auto', offset: [10, 0], opacity: isTop ? 1 : 0.75, className: 'bg-transparent border-0 shadow-none' }
      )

      marker.addTo(map)
      markersRef.current.set(vessel.mmsi, marker)
    })

    return () => {
      markersRef.current.forEach(m => m.remove())
      markersRef.current.clear()
    }
  }, [vessels, selectedMMSI, scrubIdx, map, onVesselClick])

  // Track polylines via react-leaflet (these are fine to re-render)
  return (
    <>
      {vessels.map(vessel => {
        const coords     = parseTrackCoords(vessel.track_geojson)
        if (coords.length < 2) return null
        const isSelected = vessel.mmsi === selectedMMSI
        const isTop      = vessel.rank === 1
        const color      = scoreColor(vessel.total_score, isSelected, isTop)
        const visible    = isSelected ? coords.slice(0, Math.max(2, scrubIdx + 1)) : coords

        return (
          <Polyline
            key={vessel.mmsi + '-trk'}
            positions={visible}
            pathOptions={{
              color,
              weight:    isSelected ? 2.5 : isTop ? 2 : 1,
              opacity:   isSelected ? 0.9 : isTop ? 0.65 : 0.3,
              dashArray: isSelected ? undefined : '3 5',
            }}
          />
        )
      })}
    </>
  )
}
