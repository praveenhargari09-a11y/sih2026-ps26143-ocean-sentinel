/**
 * MapView — full-bleed Leaflet map. Hosts the scrubber overlay outside
 * MapContainer so it can be positioned over the map without being inside
 * the Leaflet context.
 */
import React, { useState, useCallback, useEffect, useRef } from 'react'
import { MapContainer, TileLayer, useMap } from 'react-leaflet'
import { Play, Pause, SkipBack } from 'lucide-react'
import type { SpillRecord, DriftResult, VesselInfo, LayerVisibility } from '../types'
import SpillLayer from './SpillLayer'
import DriftLayer from './DriftLayer'
import VesselLayer, { parseTrackCoords } from './VesselLayer'

interface Props {
  spill: SpillRecord | null
  drift: DriftResult | null
  vessels: VesselInfo[]
  selectedVesselMMSI: string | null
  onVesselClick: (mmsi: string | null) => void
  layers: LayerVisibility
}

function MapAutoPan({ spill }: { spill: SpillRecord | null }) {
  const map   = useMap()
  const prevId = useRef<string | null>(null)
  useEffect(() => {
    if (spill && spill.id !== prevId.current && spill.centroid_lat != null && spill.centroid_lon != null) {
      prevId.current = spill.id
      map.flyTo([spill.centroid_lat, spill.centroid_lon], 7, { duration: 1.2 })
    }
  }, [spill, map])
  return null
}

// ── Scrubber overlay (rendered OUTSIDE MapContainer) ───────────────────────
function ScrubberOverlay({
  vessel, coords, scrubIdx, setScrubIdx, playing, setPlaying, onClose,
}: {
  vessel: VesselInfo
  coords: [number, number][]
  scrubIdx: number
  setScrubIdx: (i: number) => void
  playing: boolean
  setPlaying: (p: boolean) => void
  onClose: () => void
}) {
  const pct = coords.length > 1 ? Math.round((scrubIdx / (coords.length - 1)) * 100) : 0

  return (
    <div className="absolute bottom-10 left-1/2 -translate-x-1/2 z-[500] pointer-events-auto">
      <div
        className="flex flex-col gap-2 px-4 py-2.5 rounded shadow-2xl min-w-[280px]"
        style={{ background: '#161b22', border: '1px solid #30363d' }}
      >
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-2">
            <span
              className="data-mono text-[11px]"
              style={{ color: '#22d3ee' }}
            >{vessel.mmsi}</span>
            <span className="text-[11px]" style={{ color: '#8b949e' }}>
              {vessel.vessel_name}
            </span>
          </div>
          <button
            onClick={onClose}
            className="text-[#8b949e] hover:text-[#e6edf3] text-[11px] ml-4"
          >✕</button>
        </div>
        <div className="flex items-center gap-2">
          <button
            onClick={() => { setScrubIdx(0); setPlaying(false) }}
            className="text-[#8b949e] hover:text-[#e6edf3]"
          ><SkipBack size={12} /></button>
          <button
            onClick={() => setPlaying(!playing)}
            className="text-[#fbbf24] hover:text-[#fcd34d]"
          >{playing ? <Pause size={13} /> : <Play size={13} />}</button>
          <input
            type="range"
            min={0}
            max={Math.max(1, coords.length - 1)}
            value={scrubIdx}
            onChange={e => { setScrubIdx(Number(e.target.value)); setPlaying(false) }}
            className="scrubber flex-1"
          />
          <span
            className="data-mono text-[10px] w-8 text-right shrink-0"
            style={{ color: '#8b949e' }}
          >{pct}%</span>
        </div>
        <div
          className="flex justify-between text-[10px]"
          style={{ color: '#484f58' }}
        >
          <span>Track origin</span>
          <span className="data-mono">{coords.length} pts</span>
          <span>Latest</span>
        </div>
      </div>
    </div>
  )
}

// ── Main export ────────────────────────────────────────────────────────────
export default function MapView({
  spill, drift, vessels, selectedVesselMMSI, onVesselClick, layers,
}: Props) {
  const [scrubIdx, setScrubIdx]   = useState(0)
  const [playing, setPlayingState] = useState(false)
  const playRef  = useRef(false)
  const animRef  = useRef<number>()

  const selectedVessel = vessels.find(v => v.mmsi === selectedVesselMMSI) ?? null
  const scrubCoords    = selectedVessel ? parseTrackCoords(selectedVessel.track_geojson) : []

  const setPlaying = useCallback((v: boolean) => {
    playRef.current = v
    setPlayingState(v)
  }, [])

  // Reset on vessel change
  useEffect(() => { setScrubIdx(0); setPlaying(false) }, [selectedVesselMMSI, setPlaying])

  // Playback loop
  useEffect(() => {
    if (!playing || scrubCoords.length < 2) return
    let last = performance.now()
    const step = (now: number) => {
      if (!playRef.current) return
      if (now - last > 75) {
        last = now
        setScrubIdx(prev => {
          if (prev >= scrubCoords.length - 1) { setPlaying(false); return prev }
          return prev + 1
        })
      }
      animRef.current = requestAnimationFrame(step)
    }
    animRef.current = requestAnimationFrame(step)
    return () => { if (animRef.current) cancelAnimationFrame(animRef.current) }
  }, [playing, scrubCoords.length, setPlaying])

  return (
    <div className="w-full h-full relative" style={{ background: '#0d1117' }}>
      <MapContainer
        center={[20, 68]}
        zoom={5}
        style={{ width: '100%', height: '100%', background: '#0d1117' }}
        zoomControl={true}
        attributionControl={true}
      >
        <TileLayer
          url="https://server.arcgisonline.com/ArcGIS/rest/services/Canvas/World_Dark_Gray_Base/MapServer/tile/{z}/{y}/{x}"
          attribution='Tiles &copy; <a href="https://www.esri.com/">Esri</a>'
          maxZoom={16}
        />

        <MapAutoPan spill={spill} />

        {(layers.hindcast || layers.forecast) && (
          <DriftLayer drift={drift} showHindcast={layers.hindcast} showForecast={layers.forecast} />
        )}

        {layers.spillPolygon && (
          <SpillLayer spill={spill} drift={drift} showUncertainty={layers.uncertaintyCircle} />
        )}

        {layers.vesselTracks && (
          <VesselLayer
            vessels={vessels}
            selectedMMSI={selectedVesselMMSI}
            onVesselClick={onVesselClick}
            scrubIdx={scrubIdx}
          />
        )}
      </MapContainer>

      {/* Scrubber overlay — outside MapContainer, inside map wrapper div */}
      {selectedVessel && scrubCoords.length > 1 && (
        <ScrubberOverlay
          vessel={selectedVessel}
          coords={scrubCoords}
          scrubIdx={scrubIdx}
          setScrubIdx={setScrubIdx}
          playing={playing}
          setPlaying={setPlaying}
          onClose={() => onVesselClick(null)}
        />
      )}
    </div>
  )
}
