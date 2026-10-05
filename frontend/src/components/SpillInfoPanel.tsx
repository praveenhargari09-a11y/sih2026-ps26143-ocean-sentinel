/**
 * SpillInfoPanel — dense information display for the selected spill event.
 * Shows detection geometry, age estimate, drift origin, and quick-action buttons.
 */
import React from 'react'
import {
  MapPin, Clock, Layers, Wind, Navigation,
  AlertTriangle, Loader2, ChevronRight,
} from 'lucide-react'
import type { SpillRecord, DriftResult } from '../types'

interface Props {
  spill: SpillRecord | null
  drift: DriftResult | null
  onRunDrift: () => Promise<void>
  isLoading: boolean
}

// ── Utility ────────────────────────────────────────────────────────────────
function formatCoord(v: number, axis: 'lat' | 'lon'): string {
  const abs = Math.abs(v).toFixed(4)
  const dir = axis === 'lat' ? (v >= 0 ? 'N' : 'S') : (v >= 0 ? 'E' : 'W')
  return `${abs}°${dir}`
}

function formatTime(iso?: string): string {
  if (!iso) return '—'
  try {
    return new Date(iso).toLocaleString('en-GB', {
      day: '2-digit', month: 'short', year: 'numeric',
      hour: '2-digit', minute: '2-digit', timeZone: 'UTC',
    }) + ' Z'
  } catch { return iso }
}

function statusColor(status: string): string {
  switch (status) {
    case 'complete':    return '#3fb950'
    case 'detecting':
    case 'drifting':
    case 'attributing': return '#f59e0b'
    case 'error':       return '#f85149'
    default:            return '#8b949e'
  }
}

// ── Row component ──────────────────────────────────────────────────────────
function InfoRow({
  icon, label, value, mono = false, accent = false,
}: {
  icon: React.ReactNode; label: string; value: React.ReactNode; mono?: boolean; accent?: boolean
}) {
  return (
    <div className="flex items-start gap-2.5 py-2 border-b border-ops-border/60">
      <div className="w-3.5 shrink-0 mt-0.5 text-ops-muted">{icon}</div>
      <div className="section-label w-28 shrink-0 pt-0.5">{label}</div>
      <div className={`flex-1 text-right ${mono ? 'data-mono' : ''} ${accent ? 'text-amber-400' : 'text-[#e6edf3]'} text-[12px]`}>
        {value}
      </div>
    </div>
  )
}

// ── Empty state ────────────────────────────────────────────────────────────
function EmptyState() {
  return (
    <div className="flex flex-col items-center justify-center h-56 text-center px-6">
      <Layers size={28} className="text-ops-border mb-3" />
      <div className="text-[13px] text-ops-muted mb-1">No spill selected</div>
      <div className="text-[11px] text-ops-border">
        Click "Demo Scenario" to load a sample event, or upload a SAR scene
      </div>
    </div>
  )
}

// ── Main component ─────────────────────────────────────────────────────────
export default function SpillInfoPanel({ spill, drift, onRunDrift, isLoading }: Props) {
  if (!spill) return <EmptyState />

  const age = spill.age_estimate
  const hasDrift = !!drift

  return (
    <div className="flex flex-col">
      {/* Header */}
      <div className="px-3 pt-3 pb-2 border-b border-ops-border">
        <div className="flex items-start justify-between gap-2">
          <div>
            <div className="text-[13px] font-semibold text-[#e6edf3] leading-tight">{spill.name}</div>
            <div className="data-mono text-[10px] text-ops-muted mt-0.5">{spill.id.substring(0, 8)}</div>
          </div>
          <span
            className="text-[10px] font-semibold px-2 py-0.5 rounded border shrink-0 mt-0.5"
            style={{
              color:       statusColor(spill.status),
              borderColor: statusColor(spill.status) + '50',
              background:  statusColor(spill.status) + '15',
            }}
          >
            {spill.status.toUpperCase()}
          </span>
        </div>
      </div>

      {/* Detection geometry */}
      <div className="px-3 py-1">
        <div className="section-label py-2">Detection</div>
        <InfoRow
          icon={<Layers size={11} />}
          label="Area"
          value={<><span className="data-mono">{spill.area_km2?.toFixed(2) ?? '—'}</span> km²</>}
          mono
        />
        <InfoRow
          icon={<MapPin size={11} />}
          label="Centroid"
          value={`${formatCoord(spill.centroid_lat, 'lat')}  ${formatCoord(spill.centroid_lon, 'lon')}`}
          mono
        />
        <InfoRow
          icon={<Clock size={11} />}
          label="Acquired"
          value={formatTime(spill.acquisition_time)}
          mono
        />
        {spill.detection_time && (
          <InfoRow
            icon={<Clock size={11} />}
            label="Processed"
            value={formatTime(spill.detection_time)}
            mono
          />
        )}
      </div>

      {/* Age estimate */}
      {age && (
        <div className="px-3 py-1 border-t border-ops-border/60">
          <div className="section-label py-2">Age Estimate (Fay Spreading)</div>
          <InfoRow
            icon={<Clock size={11} />}
            label="Min age"
            value={`${age.age_hours_min.toFixed(1)} h`}
            mono
          />
          <InfoRow
            icon={<Clock size={11} />}
            label="Max age"
            value={`${age.age_hours_max.toFixed(1)} h`}
            mono
          />
          <InfoRow
            icon={<Clock size={11} />}
            label="Mean"
            value={`${age.age_hours_mean.toFixed(1)} h`}
            mono
            accent
          />
          <InfoRow
            icon={<AlertTriangle size={11} />}
            label="Method"
            value={age.method}
          />
        </div>
      )}

      {/* Drift origin */}
      {drift ? (
        <div className="px-3 py-1 border-t border-ops-border/60">
          <div className="section-label py-2">Drift Origin</div>
          <InfoRow
            icon={<MapPin size={11} />}
            label="Position"
            value={`${formatCoord(drift.origin_lat, 'lat')}  ${formatCoord(drift.origin_lon, 'lon')}`}
            mono
          />
          <InfoRow
            icon={<Clock size={11} />}
            label="Est. time"
            value={formatTime(drift.origin_time)}
            mono
          />
          <InfoRow
            icon={<Navigation size={11} />}
            label="Uncertainty"
            value={<><span className="data-mono text-amber-400">{drift.uncertainty_km?.toFixed(1) ?? '—'}</span> km</>}
          />
          {drift.confidence !== undefined && (
            <div className="flex items-center gap-2 py-2 border-b border-ops-border/60">
              <div className="w-3.5 shrink-0"><Wind size={11} className="text-ops-muted" /></div>
              <div className="section-label w-28 shrink-0">Confidence</div>
              <div className="flex-1 flex justify-end items-center gap-2">
                <div className="score-bar-track w-20">
                  <div
                    className="score-bar-fill"
                    style={{ width: `${(drift.confidence * 100).toFixed(0)}%`, background: '#3fb950' }}
                  />
                </div>
                <span className="data-mono text-[12px] text-success">{(drift.confidence * 100).toFixed(0)}</span>
              </div>
            </div>
          )}
          {drift.warnings && drift.warnings.length > 0 && (
            <div className="py-2 space-y-1.5 border-b border-ops-border/60">
              {drift.warnings.map((warn, i) => (
                <div key={i} className="flex items-start gap-2 p-2 bg-amber-500/10 border border-amber-400/30 rounded">
                  <AlertTriangle size={12} className="text-amber-400 shrink-0 mt-0.5" />
                  <span className="text-[10px] leading-tight text-amber-200">
                    {warn}
                  </span>
                </div>
              ))}
            </div>
          )}
        </div>
      ) : (
        <div className="px-3 border-t border-ops-border/60">
          <div className="section-label py-2">Drift Analysis</div>
          <div className="py-3 text-center text-[11px] text-ops-muted mb-1">
            Run the drift model to calculate spill origin and trajectory
          </div>
          <button
            onClick={onRunDrift}
            disabled={isLoading}
            className="w-full flex items-center justify-center gap-2 py-1.5 mb-3 text-[12px]
                       font-medium rounded border transition-colors
                       bg-amber-500/10 border-amber-400/30 text-amber-400
                       hover:bg-amber-500/20 hover:border-amber-400/50
                       disabled:opacity-40"
          >
            {isLoading
              ? <><Loader2 size={13} className="animate-spin" /> Running…</>
              : <><ChevronRight size={13} /> Run Drift Model</>
            }
          </button>
        </div>
      )}

      {/* Run drift again button when drift exists */}
      {hasDrift && (
        <div className="px-3 py-2 border-t border-ops-border/60">
          <button
            onClick={onRunDrift}
            disabled={isLoading}
            className="w-full py-1 text-[11px] text-ops-muted border border-ops-border rounded
                       hover:border-ops-muted hover:text-[#e6edf3] disabled:opacity-40 transition-colors"
          >
            {isLoading
              ? <span className="flex items-center justify-center gap-2"><Loader2 size={11} className="animate-spin" /> Recalculating…</span>
              : 'Re-run Drift Model'
            }
          </button>
        </div>
      )}
    </div>
  )
}
