/**
 * VesselRankingPanel — the primary analyst interface.
 * Ranked table with score bars, expandable Recharts breakdown,
 * and distinct visual treatment for the top suspect.
 */
import React, { useState } from 'react'
import {
  BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer, Cell,
} from 'recharts'
import { ChevronDown, ChevronUp, AlertTriangle, Anchor, Ship } from 'lucide-react'
import type { VesselInfo, ScoreBreakdown } from '../types'

interface Props {
  vessels: VesselInfo[]
  selectedMMSI: string | null
  onVesselClick: (mmsi: string | null) => void
  onLoadVessels: () => Promise<void>
  isLoading: boolean
}

// ── Normalise score breakdown keys ─────────────────────────────────────────
function normaliseBreakdown(sb: ScoreBreakdown): { key: string; label: string; value: number }[] {
  return [
    { key: 'proximity',    label: 'Proximity',    value: sb.proximity    ?? sb.temporal         ?? 0 },
    { key: 'trajectory',   label: 'Trajectory',   value: sb.trajectory   ?? sb.course_alignment ?? 0 },
    { key: 'darkness',     label: 'AIS Dark',     value: sb.darkness     ?? sb.ais_gap          ?? 0 },
    { key: 'vessel_type',  label: 'Vessel Type',  value: sb.vessel_type  ?? sb.historical       ?? 0 },
    { key: 'speed_anomaly',label: 'Speed Anomaly',value: sb.speed_anomaly ?? 0 },
  ]
}

// ── Score colour ───────────────────────────────────────────────────────────
function scoreColor(s: number): string {
  if (s >= 0.75) return '#f85149'
  if (s >= 0.5)  return '#f59e0b'
  if (s >= 0.25) return '#fbbf24'
  return '#4b5563'
}

function scoreBg(s: number): string {
  if (s >= 0.75) return 'rgba(248,81,73,0.12)'
  if (s >= 0.5)  return 'rgba(245,158,11,0.12)'
  return 'transparent'
}

// ── Recharts custom tooltip ────────────────────────────────────────────────
function ChartTooltip({ active, payload }: any) {
  if (!active || !payload?.length) return null
  const { label, value } = payload[0].payload
  return (
    <div className="bg-ops-panel border border-ops-border rounded px-2 py-1 text-[11px]">
      <div className="text-ops-muted">{label}</div>
      <div className="data-mono text-amber-400">{(value * 100).toFixed(0)}</div>
    </div>
  )
}

// ── Vessel row ─────────────────────────────────────────────────────────────
function VesselRow({
  vessel, isSelected, isExpanded, onSelect, onExpand,
}: {
  vessel: VesselInfo
  isSelected: boolean
  isExpanded: boolean
  onSelect: () => void
  onExpand: (e: React.MouseEvent) => void
}) {
  const isTop      = vessel.rank === 1
  const color      = scoreColor(vessel.total_score)
  const scorePct   = vessel.total_score * 100
  const breakdown  = normaliseBreakdown(vessel.score_breakdown)
  const hasGaps    = vessel.ais_gaps?.length > 0

  return (
    <div
      className={`vessel-row ${isSelected ? 'selected' : ''} ${isTop ? 'top-suspect' : ''}`}
      style={isTop ? { borderLeftColor: '#fbbf24' } : undefined}
    >
      {/* Main row */}
      <div
        className="flex items-center gap-2 px-3 py-2 cursor-pointer"
        onClick={onSelect}
      >
        {/* Rank */}
        <div className={`w-5 h-5 rounded flex items-center justify-center text-[10px] font-bold shrink-0
                         data-mono ${isTop ? 'bg-amber-400/15 text-amber-400' : 'bg-ops-subtle text-ops-muted'}`}>
          {vessel.rank}
        </div>

        {/* Identity */}
        <div className="flex-1 min-w-0">
          <div className="flex items-center gap-1.5">
            <span className="text-[12px] font-medium text-[#e6edf3] truncate">{vessel.vessel_name || 'Unknown'}</span>
            {hasGaps && (
              <span className="shrink-0 flex items-center gap-0.5 px-1 py-0 text-[9px] font-semibold
                               bg-danger/10 border border-danger/30 text-danger rounded">
                <AlertTriangle size={7} />AIS GAP
              </span>
            )}
          </div>
          <div className="flex items-center gap-1.5 mt-0.5">
            <span className="data-mono text-cyan-400 text-[10px]">{vessel.mmsi}</span>
            <span className="text-ops-border">·</span>
            <span className="text-ops-muted text-[10px]">{vessel.vessel_type}</span>
            <span className="text-ops-border">·</span>
            <span className="text-ops-muted text-[10px]">{vessel.flag}</span>
          </div>
        </div>

        {/* Score */}
        <div className="flex flex-col items-end gap-1 shrink-0 w-14">
          <span className="data-mono text-[13px] font-bold" style={{ color }}>
            {scorePct.toFixed(0)}
          </span>
          <div className="score-bar-track w-14">
            <div className="score-bar-fill" style={{ width: `${scorePct}%`, background: color }} />
          </div>
        </div>

        {/* Expand chevron */}
        <button
          onClick={onExpand}
          className="text-ops-muted hover:text-[#e6edf3] shrink-0 ml-1"
        >
          {isExpanded ? <ChevronUp size={13} /> : <ChevronDown size={13} />}
        </button>
      </div>

      {/* Expanded: 5-criterion Recharts BarChart */}
      {isExpanded && (
        <div className="px-3 pb-3 pt-0">
          <div className="border-t border-ops-border pt-2">
            {/* Score bars (text) */}
            <div className="grid grid-cols-5 gap-1 mb-3">
              {breakdown.map(({ label, value }) => (
                <div key={label} className="text-center">
                  <div className="text-[8px] section-label mb-1">{label}</div>
                  <div className="data-mono text-[11px] font-bold" style={{ color: scoreColor(value) }}>
                    {(value * 100).toFixed(0)}
                  </div>
                  <div className="score-bar-track mt-1">
                    <div className="score-bar-fill" style={{ width: `${value * 100}%`, background: scoreColor(value) }} />
                  </div>
                </div>
              ))}
            </div>

            {/* Recharts mini bar chart */}
            <ResponsiveContainer width="100%" height={72}>
              <BarChart data={breakdown.map(b => ({ ...b, display: (b.value * 100).toFixed(0) }))}
                        margin={{ top: 0, right: 0, bottom: 0, left: -32 }}
                        barCategoryGap="20%">
                <XAxis
                  dataKey="label"
                  tick={{ fill: '#6e7681', fontSize: 9, fontFamily: 'Inter' }}
                  tickLine={false}
                  axisLine={false}
                />
                <YAxis domain={[0, 1]} tick={false} axisLine={false} tickLine={false} />
                <Tooltip content={<ChartTooltip />} cursor={{ fill: 'rgba(48,54,61,0.4)' }} />
                <Bar dataKey="value" radius={[2, 2, 0, 0]} maxBarSize={24}>
                  {breakdown.map((b, i) => (
                    <Cell key={i} fill={scoreColor(b.value)} fillOpacity={0.8} />
                  ))}
                </Bar>
              </BarChart>
            </ResponsiveContainer>

            {/* AIS gaps detail */}
            {hasGaps && (
              <div className="mt-2 border-t border-ops-border pt-2">
                <div className="section-label mb-1.5">AIS Gaps Detected</div>
                {vessel.ais_gaps.slice(0, 3).map((gap, i) => {
                  const start = gap.start ?? gap.gap_start ?? '—'
                  const hrs   = gap.duration_hours ?? ((gap.duration_minutes ?? 0) / 60)
                  return (
                    <div key={i} className="flex items-center gap-2 py-0.5">
                      <div className="w-1 h-1 rounded-full bg-danger shrink-0" />
                      <span className="data-mono text-[10px] text-danger/80">{hrs.toFixed(1)}h</span>
                      <span className="data-mono text-[10px] text-ops-muted truncate">{start}</span>
                    </div>
                  )
                })}
              </div>
            )}

            {/* Proximity */}
            <div className="mt-2 flex items-center gap-2 text-[10px] text-ops-muted">
              <span>Closest approach:</span>
              <span className="data-mono text-[#e6edf3]">{vessel.proximity_km.toFixed(1)} km</span>
            </div>
          </div>
        </div>
      )}
    </div>
  )
}

// ── Empty state ────────────────────────────────────────────────────────────
function EmptyState({ onLoad, isLoading }: { onLoad: () => void; isLoading: boolean }) {
  return (
    <div className="flex flex-col items-center justify-center h-64 px-6 text-center">
      <Anchor size={28} className="text-ops-border mb-3" />
      <div className="text-[13px] text-ops-muted mb-1">No vessels attributed</div>
      <div className="text-[11px] text-ops-border mb-4">
        Run the AIS correlation pipeline to identify suspect vessels
      </div>
      <button
        onClick={onLoad}
        disabled={isLoading}
        className="flex items-center gap-2 px-3 py-1.5 text-[12px] font-medium rounded
                   bg-amber-400/10 border border-amber-400/30 text-amber-400
                   hover:bg-amber-400/20 hover:border-amber-400/50
                   disabled:opacity-40 transition-colors"
      >
        {isLoading ? (
          <span className="w-3 h-3 border border-amber-400 border-t-transparent rounded-full animate-spin" />
        ) : (
          <Ship size={13} />
        )}
        Run Attribution
      </button>
    </div>
  )
}

// ── Main panel ─────────────────────────────────────────────────────────────
export default function VesselRankingPanel({
  vessels, selectedMMSI, onVesselClick, onLoadVessels, isLoading,
}: Props) {
  const [expandedMMSI, setExpandedMMSI] = useState<string | null>(null)

  const handleSelect = (mmsi: string) => {
    onVesselClick(mmsi === selectedMMSI ? null : mmsi)
  }

  const handleExpand = (e: React.MouseEvent, mmsi: string) => {
    e.stopPropagation()
    setExpandedMMSI(prev => prev === mmsi ? null : mmsi)
  }

  if (!vessels.length) {
    return <EmptyState onLoad={onLoadVessels} isLoading={isLoading} />
  }

  // Legend row
  const topScore  = vessels[0].total_score
  const topColor  = scoreColor(topScore)

  return (
    <div className="flex flex-col">
      {/* Summary strip */}
      <div className="flex items-center justify-between px-3 py-2 border-b border-ops-border bg-ops-subtle/50">
        <span className="section-label">{vessels.length} Candidates Ranked</span>
        <div className="flex items-center gap-1.5">
          <span className="text-[10px] text-ops-muted">Top score:</span>
          <span className="data-mono font-bold text-[11px]" style={{ color: topColor }}>
            {(topScore * 100).toFixed(0)}
          </span>
        </div>
      </div>

      {/* Top suspect callout */}
      {vessels[0] && (
        <div
          className="mx-3 mt-2 mb-1 px-3 py-2 rounded border cursor-pointer transition-colors"
          style={{
            background: scoreBg(vessels[0].total_score),
            borderColor: `${topColor}50`,
          }}
          onClick={() => handleSelect(vessels[0].mmsi)}
        >
          <div className="flex items-center justify-between">
            <div className="section-label text-amber-400">TOP SUSPECT</div>
            <span className="data-mono font-bold" style={{ color: topColor }}>
              {(vessels[0].total_score * 100).toFixed(0)}
            </span>
          </div>
          <div className="text-[13px] font-semibold text-[#e6edf3] mt-0.5">{vessels[0].vessel_name}</div>
          <div className="flex items-center gap-2 mt-1 text-[10px] text-ops-muted">
            <span className="data-mono text-cyan-400">{vessels[0].mmsi}</span>
            <span>·</span>
            <span>{vessels[0].vessel_type}</span>
            <span>·</span>
            <span>{vessels[0].flag}</span>
          </div>
        </div>
      )}

      {/* Ranked list */}
      <div className="mt-1">
        {vessels.map(vessel => (
          <VesselRow
            key={vessel.mmsi}
            vessel={vessel}
            isSelected={vessel.mmsi === selectedMMSI}
            isExpanded={expandedMMSI === vessel.mmsi}
            onSelect={() => handleSelect(vessel.mmsi)}
            onExpand={(e) => handleExpand(e, vessel.mmsi)}
          />
        ))}
      </div>
    </div>
  )
}
