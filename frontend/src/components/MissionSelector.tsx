/**
 * MissionSelector — military-style incident selector for the TopBar.
 * 
 * Shows real historical incidents by default.
 * "Arabian Sea Demo" only visible with ?dev=1 in URL.
 * Displays data_source badge (REAL / RECONSTRUCTED) for transparency.
 */
import React, { useState, useEffect, useRef } from 'react'
import { ChevronDown, MapPin, Shield, AlertTriangle } from 'lucide-react'

interface Incident {
  id: string
  name: string
  description: string
  date: string
  location: [number, number]
  data_source: string
  has_sar: boolean
  has_ocean: boolean
  has_ais: boolean
}

interface Props {
  onSelect: (incidentId: string) => void
  isLoading: boolean
  currentIncidentName?: string
}

function DataSourceBadge({ source }: { source: string }) {
  if (source === 'real') {
    return (
      <span className="px-1.5 py-0.5 text-[8px] font-bold tracking-wider rounded
                       bg-success/20 text-success border border-success/30">
        REAL DATA
      </span>
    )
  }
  return (
    <span className="px-1.5 py-0.5 text-[8px] font-bold tracking-wider rounded
                     bg-amber-500/15 text-amber-400 border border-amber-400/30">
      RECONSTRUCTED
    </span>
  )
}

export default function MissionSelector({ onSelect, isLoading, currentIncidentName }: Props) {
  const [incidents, setIncidents] = useState<Incident[]>([])
  const [open, setOpen] = useState(false)
  const [loaded, setLoaded] = useState(false)
  const ref = useRef<HTMLDivElement>(null)

  // Check for dev mode
  const isDev = typeof window !== 'undefined' && new URLSearchParams(window.location.search).has('dev')

  // Fetch incidents list on mount
  useEffect(() => {
    fetch('/api/incidents')
      .then(r => r.json())
      .then((data: Incident[]) => {
        // Filter out demo incidents unless in dev mode
        const filtered = isDev
          ? data
          : data.filter(i => !i.id.includes('demo') && !i.id.includes('arabian'))
        setIncidents(filtered)
        setLoaded(true)
      })
      .catch(err => {
        console.warn('[MissionSelector] Failed to load incidents:', err)
        setLoaded(true)
      })
  }, [isDev])

  // Close on outside click
  useEffect(() => {
    const handler = (e: MouseEvent) => {
      if (ref.current && !ref.current.contains(e.target as Node)) setOpen(false)
    }
    document.addEventListener('mousedown', handler)
    return () => document.removeEventListener('mousedown', handler)
  }, [])

  if (!loaded || incidents.length === 0) return null

  return (
    <div ref={ref} className="relative">
      <button
        onClick={() => setOpen(v => !v)}
        disabled={isLoading}
        className="flex items-center gap-1.5 px-2.5 py-1 text-[11px] font-medium rounded
                   bg-ops-subtle border border-ops-border text-ops-muted
                   hover:bg-ops-raised hover:text-[#e6edf3] hover:border-amber-400/40
                   disabled:opacity-40 transition-colors"
      >
        <Shield size={11} className="text-amber-400" />
        {currentIncidentName || 'Select Mission'}
        <ChevronDown size={10} className={`transition-transform ${open ? 'rotate-180' : ''}`} />
      </button>

      {open && (
        <div className="absolute top-full right-0 mt-1 w-80 max-w-[calc(100vw-1rem)] z-[900]
                        bg-ops-panel border border-ops-border rounded-lg shadow-2xl
                        overflow-hidden animate-dropdown-in">
          {/* Header */}
          <div className="px-3 py-2 border-b border-ops-border bg-ops-subtle">
            <span className="text-[10px] font-bold tracking-widest text-ops-muted uppercase">
              Mission Select
            </span>
          </div>

          {/* Incident list */}
          <div className="max-h-64 overflow-y-auto">
            {incidents.map(incident => (
              <button
                key={incident.id}
                onClick={() => {
                  onSelect(incident.id)
                  setOpen(false)
                }}
                disabled={isLoading}
                className="w-full text-left px-3 py-2.5 border-b border-ops-border/50
                          hover:bg-ops-raised transition-colors disabled:opacity-40
                          group"
              >
                <div className="flex items-start justify-between gap-2">
                  <div className="flex-1 min-w-0">
                    <div className="flex items-center gap-2">
                      <span className="text-[12px] font-medium text-[#e6edf3] group-hover:text-amber-400 transition-colors">
                        {incident.name}
                      </span>
                    </div>
                    <div className="flex items-center gap-2 mt-0.5">
                      <span className="text-[10px] text-ops-muted">
                        {incident.date ? new Date(incident.date).toLocaleDateString('en-US', {
                          year: 'numeric', month: 'short', day: 'numeric'
                        }) : ''}
                      </span>
                      <DataSourceBadge source={incident.data_source} />
                    </div>
                    {incident.description && (
                      <p className="text-[10px] text-ops-muted mt-1 truncate">
                        {incident.description}
                      </p>
                    )}
                  </div>
                  <MapPin size={12} className="text-ops-muted mt-1 shrink-0" />
                </div>
              </button>
            ))}
          </div>

          {/* Footer */}
          <div className="px-3 py-1.5 bg-ops-subtle border-t border-ops-border">
            <span className="text-[9px] text-ops-muted">
              {incidents.length} incident{incidents.length !== 1 ? 's' : ''} available
              {isDev && <span className="text-amber-400 ml-1">• DEV MODE</span>}
            </span>
          </div>
        </div>
      )}
    </div>
  )
}
