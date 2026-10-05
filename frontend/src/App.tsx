import React, { useState, useCallback, createContext, useContext } from 'react'
import MapView from './components/MapView'
import VesselRankingPanel from './components/VesselRankingPanel'
import SpillInfoPanel from './components/SpillInfoPanel'
import PipelineStatusBar from './components/PipelineStatusBar'
import UploadPanel from './components/UploadPanel'
import MissionSelector from './components/MissionSelector'
import { useSpillData } from './hooks/useSpillData'
import { useWebSocket } from './hooks/useWebSocket'
import type { LayerVisibility, RightPanelTab } from './types'
import {
  ChevronRight, ChevronLeft, Layers, Upload,
  Radio, AlertTriangle, Ship
} from 'lucide-react'

import { LayerContext, useLayerContext } from './context/LayerContext'

import DataSourceBadge from './components/DataSourceBadge'

// ── Topbar ─────────────────────────────────────────────────────────────────
function TopBar({
  spillName, acquisitionTime, isConnected, onUpload, onDemo, onIncident, isLoading,
  aisSource, sarSource
}: {
  spillName?: string; acquisitionTime?: string; isConnected: boolean;
  onUpload: () => void; onDemo: () => void; onIncident: (id: string) => void; isLoading: boolean;
  aisSource?: string; sarSource?: string;
}) {
  return (
    <header className="flex items-center justify-between px-3 h-14 bg-ops-panel border-b border-ops-border shrink-0 z-[950]">
      {/* Brand */}
      <div className="flex items-center gap-2 shrink-0">
        <span className="text-amber-400 text-base">⬡</span>
        <span className="font-semibold text-[13px] tracking-tight text-[#e6edf3]">Ocean Sentinel</span>
        <span className="text-ops-border mx-1">|</span>
        <span className="section-label">Maritime Intelligence</span>
      </div>

      {/* Scene info & Badges */}
      {spillName && (
        <div className="flex flex-col items-center justify-center absolute left-1/2 -translate-x-1/2 mt-1">
          <div className="flex items-center gap-3">
            <span className="text-[#e6edf3] text-[13px] font-medium truncate max-w-64">{spillName}</span>
            {acquisitionTime && (
              <span className="data-mono text-cyan-400 text-[11px]">
                {new Date(acquisitionTime).toUTCString().replace('GMT', 'Z').substring(5, 22)}
              </span>
            )}
          </div>
          <div className="mt-1">
            <DataSourceBadge aisSource={aisSource} sarSource={sarSource} />
          </div>
        </div>
      )}

      {/* Actions + status */}
      <div className="flex items-center gap-2 shrink-0">
        <MissionSelector
          onSelect={onIncident}
          isLoading={isLoading}
          currentIncidentName={spillName}
        />
        <button
          onClick={onDemo}
          disabled={isLoading}
          className="flex items-center gap-1.5 px-2.5 py-1 text-[11px] font-medium rounded
                     bg-ops-subtle border border-ops-border text-ops-muted
                     hover:bg-ops-raised hover:text-[#e6edf3] hover:border-amber-400/40
                     disabled:opacity-40 transition-colors"
        >
          {isLoading ? (
            <span className="inline-block w-3 h-3 border border-ops-muted border-t-amber-400 rounded-full animate-spin" />
          ) : (
            <Ship size={11} />
          )}
          Demo
        </button>
        <button
          onClick={onUpload}
          className="flex items-center gap-1.5 px-2.5 py-1 text-[11px] font-medium rounded
                     bg-amber-500/10 border border-amber-400/30 text-amber-400
                     hover:bg-amber-500/20 hover:border-amber-400/60 transition-colors"
        >
          <Upload size={11} />
          Upload SAR
        </button>
        {/* Live indicator */}
        <div className="flex items-center gap-1.5 ml-1">
          <div className={`w-1.5 h-1.5 rounded-full ${isConnected ? 'bg-success' : 'bg-ops-muted'}`} />
          <span className={`text-[10px] ${isConnected ? 'text-success' : 'text-ops-muted'}`}>
            {isConnected ? 'LIVE' : 'OFFLINE'}
          </span>
        </div>
      </div>
    </header>
  )
}

// ── Right panel tab header ─────────────────────────────────────────────────
const TABS: { id: RightPanelTab; label: string; icon: React.ReactNode }[] = [
  { id: 'vessels',  label: 'Vessels',  icon: <Ship size={12} /> },
  { id: 'spill',    label: 'Spill',    icon: <AlertTriangle size={12} /> },
  { id: 'pipeline', label: 'Pipeline', icon: <Radio size={12} /> },
]

// ── Layer toggle widget ────────────────────────────────────────────────────
function LayerToggle() {
  const { layers, toggle } = useLayerContext()
  const [open, setOpen] = useState(false)

  const items: { key: keyof LayerVisibility; label: string; color: string }[] = [
    { key: 'spillPolygon',      label: 'Spill Polygon',    color: '#f85149' },
    { key: 'hindcast',          label: 'Hindcast Track',   color: '#4b5563' },
    { key: 'forecast',          label: 'Forecast Track',   color: '#f59e0b' },
    { key: 'vesselTracks',      label: 'Vessel Tracks',    color: '#22d3ee' },
    { key: 'uncertaintyCircle', label: 'Uncertainty Zone', color: '#8b949e' },
  ]

  return (
    <div className="absolute bottom-12 left-3 z-[800]">
      <button
        onClick={() => setOpen(v => !v)}
        className="flex items-center gap-1.5 px-2.5 py-1.5 text-[11px] font-medium rounded
                   bg-ops-panel border border-ops-border text-ops-muted
                   hover:text-[#e6edf3] hover:border-ops-muted transition-colors shadow-lg"
      >
        <Layers size={12} />
        Layers
      </button>
      {open && (
        <div className="absolute bottom-8 left-0 w-48 bg-ops-panel border border-ops-border rounded shadow-xl p-2 space-y-1">
          {items.map(({ key, label, color }) => (
            <label key={key} className="flex items-center gap-2 px-1 py-1 rounded hover:bg-ops-raised cursor-pointer">
              <input
                type="checkbox"
                checked={layers[key]}
                onChange={() => toggle(key)}
                className="hidden"
              />
              <span
                className={`w-3 h-3 rounded-sm border flex items-center justify-center shrink-0 transition-colors`}
                style={{
                  borderColor: layers[key] ? color : '#30363d',
                  background:  layers[key] ? color + '33' : 'transparent',
                }}
              >
                {layers[key] && <span style={{ color }} className="text-[8px] leading-none">✓</span>}
              </span>
              <span className="text-[11px] text-ops-muted">{label}</span>
              <span className="ml-auto w-3 h-1 rounded" style={{ background: color }} />
            </label>
          ))}
        </div>
      )}
    </div>
  )
}

// ── Main App ───────────────────────────────────────────────────────────────
export function App() {
  const { spills: _spills, selectedSpill, driftResult, vessels, selectedVesselMMSI,
          isLoading, error, incidentMetadata, actions } = useSpillData()
  const { status: rawWsStatus, isConnected } = useWebSocket(selectedSpill?.id ?? null)
  const wsStatus = selectedSpill?.status === 'detecting' 
    ? rawWsStatus 
    : (selectedSpill ? { stage: 'ranking', percent: 100, message: 'Ready' } : null)
  const [showUpload, setShowUpload] = useState(false)
  const [panelOpen, setPanelOpen] = useState(true)
  const [activeTab, setActiveTab] = useState<RightPanelTab>('vessels')
  const [layers, setLayers] = useState<LayerVisibility>({
    spillPolygon: true, hindcast: true, forecast: true,
    vesselTracks: true, uncertaintyCircle: true,
  })

  const toggle = useCallback((k: keyof LayerVisibility) =>
    setLayers(prev => ({ ...prev, [k]: !prev[k] })), [])

  return (
    <LayerContext.Provider value={{ layers, toggle }}>
      <div className="flex flex-col h-screen overflow-hidden" style={{ background: '#0d1117' }}>

        {/* ── Top bar ── */}
        <TopBar
          spillName={selectedSpill?.name}
          acquisitionTime={selectedSpill?.acquisition_time}
          isConnected={isConnected}
          onUpload={() => setShowUpload(true)}
          onDemo={actions.generateDemo}
          onIncident={actions.loadIncident}
          isLoading={isLoading}
          aisSource={incidentMetadata?.ais_source}
          sarSource={incidentMetadata?.sar_source}
        />

        {/* ── Main area: map + right panel ── */}
        <div className="flex flex-1 overflow-hidden relative">

          {/* Map fills all space */}
          <div className="flex-1 relative">
            <MapView
              spill={selectedSpill}
              drift={driftResult}
              vessels={vessels}
              selectedVesselMMSI={selectedVesselMMSI}
              onVesselClick={actions.selectVessel}
              layers={layers}
            />
            {/* Layer toggle widget — absolute over map */}
            <LayerToggle />
          </div>

          {/* ── Right panel ── */}
          <div
            className={`relative flex shrink-0 transition-all duration-200 ease-out
                       ${panelOpen ? 'w-[380px]' : 'w-0 overflow-hidden'}`}
          >
            {panelOpen && (
              <div className="w-[380px] flex flex-col bg-ops-panel border-l border-ops-border overflow-hidden animate-slide-in-right">
                {/* Tab strip */}
                <div className="flex border-b border-ops-border shrink-0">
                  {TABS.map(t => (
                    <button
                      key={t.id}
                      onClick={() => setActiveTab(t.id)}
                      className={`flex items-center gap-1.5 px-3 py-2.5 text-[11px] font-medium
                                 border-b-2 transition-colors flex-1 justify-center
                                 ${activeTab === t.id
                                   ? 'border-amber-400 text-amber-400'
                                   : 'border-transparent text-ops-muted hover:text-[#e6edf3]'
                                 }`}
                    >
                      {t.icon}
                      {t.label}
                      {t.id === 'vessels' && vessels.length > 0 && (
                        <span className="ml-1 px-1 py-0.5 text-[9px] bg-ops-subtle rounded text-ops-muted">
                          {vessels.length}
                        </span>
                      )}
                    </button>
                  ))}
                </div>

                {/* Panel content */}
                <div className="flex-1 overflow-y-auto overflow-x-hidden">
                  {activeTab === 'vessels' && (
                    <VesselRankingPanel
                      vessels={vessels}
                      selectedMMSI={selectedVesselMMSI}
                      onVesselClick={actions.selectVessel}
                      onLoadVessels={actions.loadVessels}
                      isLoading={isLoading}
                    />
                  )}
                  {activeTab === 'spill' && (
                    <SpillInfoPanel
                      spill={selectedSpill}
                      drift={driftResult}
                      onRunDrift={actions.runDrift}
                      isLoading={isLoading}
                    />
                  )}
                  {activeTab === 'pipeline' && (
                    <div className="p-4">
                      <PipelineStatusBar status={wsStatus} isConnected={isConnected} expanded />
                    </div>
                  )}
                </div>
              </div>
            )}

            {/* Collapse toggle button */}
            <button
              onClick={() => setPanelOpen(v => !v)}
              className="absolute -left-6 top-1/2 -translate-y-1/2 z-10
                         w-6 h-12 flex items-center justify-center
                         bg-ops-panel border border-ops-border border-r-0 rounded-l
                         text-ops-muted hover:text-amber-400 hover:border-amber-400/40
                         transition-colors shadow-lg"
            >
              {panelOpen ? <ChevronRight size={12} /> : <ChevronLeft size={12} />}
            </button>
          </div>
        </div>

        {/* ── Bottom pipeline strip ── */}
        <PipelineStatusBar status={wsStatus} isConnected={isConnected} />

        {/* ── Upload modal ── */}
        {showUpload && (
          <UploadPanel
            isOpen={showUpload}
            onClose={() => setShowUpload(false)}
            onUpload={async (fd) => {
              await actions.runDetection(fd)
              setShowUpload(false)
            }}
          />
        )}

        {/* ── Error toast ── */}
        {error && (
          <div className="fixed bottom-12 right-4 z-[2000] flex items-center gap-2
                          bg-ops-panel border border-danger/40 text-danger
                          px-3 py-2 rounded text-[12px] shadow-xl">
            <AlertTriangle size={13} />
            {error}
          </div>
        )}
      </div>
    </LayerContext.Provider>
  )
}

export default App
