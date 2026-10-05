/**
 * SpillListPanel — compact scene selector (kept for potential use as a dropdown).
 * Currently the demo/upload buttons are in the TopBar; this can be mounted
 * anywhere as a spill history list.
 */
import React from 'react'
import { AlertTriangle, CheckCircle, Clock, Layers } from 'lucide-react'
import type { SpillRecord } from '../types'

interface Props {
  spills: SpillRecord[]
  selectedSpillId: string | null
  onSelectSpill: (spill: SpillRecord) => void
  onNewAnalysis?: () => void
  onGenerateDemo?: () => void
  isLoading?: boolean
}

function statusIcon(status: string) {
  switch (status) {
    case 'complete':  return <CheckCircle size={10} className="text-success shrink-0" />
    case 'error':     return <AlertTriangle size={10} className="text-danger shrink-0" />
    default:          return <Clock size={10} className="text-amber-400 shrink-0" />
  }
}

function formatTime(iso?: string): string {
  if (!iso) return '—'
  try {
    return new Date(iso).toLocaleString('en-GB', {
      day: '2-digit', month: 'short', hour: '2-digit', minute: '2-digit',
    })
  } catch { return iso }
}

export default function SpillListPanel({
  spills, selectedSpillId, onSelectSpill, isLoading,
}: Props) {
  if (!spills.length) {
    return (
      <div className="flex flex-col items-center justify-center h-32 text-center px-4">
        <Layers size={20} className="text-ops-border mb-2" />
        <div className="text-[11px] text-ops-muted">No spill events loaded</div>
      </div>
    )
  }

  return (
    <div className="flex flex-col">
      <div className="px-3 py-2 border-b border-ops-border">
        <span className="section-label">Spill Events ({spills.length})</span>
      </div>
      {spills.map(spill => {
        const isSelected = spill.id === selectedSpillId
        return (
          <div
            key={spill.id}
            onClick={() => onSelectSpill(spill)}
            className={`flex items-start gap-2 px-3 py-2.5 cursor-pointer border-b border-ops-border/50
                         transition-colors ${isSelected
                           ? 'bg-amber-400/5 border-l-2 border-l-amber-400'
                           : 'hover:bg-ops-raised'}`}
          >
            <div className="mt-0.5">{statusIcon(spill.status)}</div>
            <div className="flex-1 min-w-0">
              <div className="text-[12px] font-medium text-[#e6edf3] truncate">{spill.name}</div>
              <div className="flex items-center gap-2 mt-0.5">
                <span className="data-mono text-[10px] text-cyan-400">{spill.area_km2?.toFixed(1)} km²</span>
                <span className="text-ops-border">·</span>
                <span className="text-[10px] text-ops-muted">{formatTime(spill.acquisition_time)}</span>
              </div>
            </div>
          </div>
        )
      })}
    </div>
  )
}
