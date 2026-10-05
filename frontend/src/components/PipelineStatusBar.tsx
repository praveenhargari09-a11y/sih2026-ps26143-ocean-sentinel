/**
 * PipelineStatusBar — displays real-time pipeline progress driven by WebSocket.
 *
 * Two modes:
 *   • Compact strip (default) — 40px bottom bar showing current stage + progress line
 *   • Expanded (inside panel) — step-by-step with pending/running/done/error states
 */
import React, { useMemo } from 'react'
import { CheckCircle, Loader, Circle, XCircle, Radio } from 'lucide-react'
import type { PipelineStatus } from '../types'

interface Props {
  status: PipelineStatus | null
  isConnected: boolean
  expanded?: boolean
}

interface Stage {
  id: string
  label: string
  shortLabel: string
  pctStart: number
  pctEnd: number
}

const STAGES: Stage[] = [
  { id: 'detection', label: 'SAR Detection',    shortLabel: 'Detect',   pctStart: 0,  pctEnd: 20  },
  { id: 'hindcast',  label: 'Hindcast',         shortLabel: 'Hindcast', pctStart: 20, pctEnd: 40  },
  { id: 'forecast',  label: 'Forecast',         shortLabel: 'Forecast', pctStart: 40, pctEnd: 60  },
  { id: 'ais',       label: 'AIS Correlation',  shortLabel: 'AIS',      pctStart: 60, pctEnd: 80  },
  { id: 'ranking',   label: 'Vessel Ranking',   shortLabel: 'Ranking',  pctStart: 80, pctEnd: 100 },
]

type StageState = 'pending' | 'running' | 'done' | 'error'

function getStageState(stage: Stage, pct: number, isError: boolean): StageState {
  if (isError && pct >= stage.pctStart && pct <= stage.pctEnd) return 'error'
  if (pct >= stage.pctEnd)   return 'done'
  if (pct >= stage.pctStart) return 'running'
  return 'pending'
}

function StageIcon({ state }: { state: StageState }) {
  switch (state) {
    case 'done':    return <CheckCircle size={14} className="text-success" />
    case 'running': return <Loader size={14} className="text-amber-400 animate-spin" />
    case 'error':   return <XCircle size={14} className="text-danger" />
    default:        return <Circle size={14} className="text-ops-border" />
  }
}

// ── Expanded view (inside right panel) ────────────────────────────────────
function ExpandedPipeline({ status, isConnected }: { status: PipelineStatus | null; isConnected: boolean }) {
  const pct     = status?.percent ?? 0
  const isError = status?.stage === 'error'

  return (
    <div className="flex flex-col gap-4">
      {/* Connection status */}
      <div className="flex items-center justify-between">
        <span className="section-label">Pipeline Status</span>
        <div className="flex items-center gap-1.5">
          <Radio size={11} className={isConnected ? 'text-success' : 'text-ops-muted'} />
          <span className={`text-[10px] ${isConnected ? 'text-success' : 'text-ops-muted'}`}>
            {isConnected ? 'WebSocket Live' : 'Not Connected'}
          </span>
        </div>
      </div>

      {/* Stages */}
      {STAGES.map((stage, idx) => {
        const state      = getStageState(stage, pct, isError)
        const isRunning  = state === 'running'
        const connDone   = idx < STAGES.length - 1 && pct >= STAGES[idx + 1].pctStart

        return (
          <div key={stage.id}>
            <div className="flex items-start gap-3">
              <div className="flex flex-col items-center">
                <StageIcon state={state} />
                {idx < STAGES.length - 1 && (
                  <div className={`w-px flex-1 mt-1 min-h-[20px] transition-colors duration-500
                                   ${connDone ? 'bg-success/50' : 'bg-ops-border'}`} />
                )}
              </div>
              <div className="flex-1 pb-4">
                <div className="flex items-center justify-between">
                  <span className={`text-[12px] font-medium transition-colors
                                    ${state === 'done'    ? 'text-[#e6edf3]'
                                      : state === 'running' ? 'text-amber-400'
                                      : state === 'error'   ? 'text-danger'
                                      : 'text-ops-muted'}`}>
                    {stage.label}
                  </span>
                  {state === 'running' && (
                    <span className="data-mono text-[10px] text-amber-400">
                      {Math.round(((pct - stage.pctStart) / (stage.pctEnd - stage.pctStart)) * 100)}%
                    </span>
                  )}
                  {state === 'done' && (
                    <span className="text-[10px] text-success">Complete</span>
                  )}
                </div>
                {isRunning && status?.message && (
                  <div className="text-[11px] text-ops-muted mt-0.5 truncate">{status.message}</div>
                )}
                {state === 'running' && (
                  <div className="mt-1.5 h-px bg-ops-border rounded overflow-hidden">
                    <div
                      className="h-full bg-amber-400/60 rounded transition-all duration-300"
                      style={{ width: `${((pct - stage.pctStart) / (stage.pctEnd - stage.pctStart)) * 100}%` }}
                    />
                  </div>
                )}
              </div>
            </div>
          </div>
        )
      })}

      {/* Error message */}
      {isError && status?.error && (
        <div className="flex items-start gap-2 p-2 bg-danger/10 border border-danger/30 rounded text-[11px] text-danger">
          <XCircle size={12} className="shrink-0 mt-0.5" />
          {status.error}
        </div>
      )}

      {/* Idle state */}
      {!status && (
        <div className="text-[12px] text-ops-muted text-center py-4">
          No active pipeline — trigger detection to begin
        </div>
      )}
    </div>
  )
}

// ── Compact bottom strip ───────────────────────────────────────────────────
function CompactBar({ status, isConnected }: { status: PipelineStatus | null; isConnected: boolean }) {
  const pct     = status?.percent ?? 0
  const isError = status?.stage === 'error'
  const isDone  = pct >= 100

  const activeStage = useMemo(() =>
    STAGES.find(s => pct >= s.pctStart && pct < s.pctEnd) ?? (isDone ? STAGES[STAGES.length - 1] : null),
    [pct, isDone])

  return (
    <div className="h-9 flex items-center gap-3 px-3 bg-ops-panel border-t border-ops-border shrink-0">
      {/* Steps */}
      <div className="flex items-center gap-1">
        {STAGES.map((stage, idx) => {
          const state = getStageState(stage, pct, isError)
          return (
            <React.Fragment key={stage.id}>
              <div
                className="flex items-center gap-1"
                title={stage.label}
              >
                <div className={`w-1.5 h-1.5 rounded-full transition-all duration-300
                                 ${state === 'done'    ? 'bg-success'
                                   : state === 'running' ? 'bg-amber-400 animate-pulse'
                                   : state === 'error'   ? 'bg-danger'
                                   : 'bg-ops-border'}`}
                />
                <span className={`text-[10px] hidden sm:block
                                  ${state === 'done'    ? 'text-success/70'
                                    : state === 'running' ? 'text-amber-400'
                                    : 'text-ops-border'}`}>
                  {stage.shortLabel}
                </span>
              </div>
              {idx < STAGES.length - 1 && (
                <div className={`w-4 h-px mx-0.5 transition-colors duration-500
                                 ${pct >= STAGES[idx + 1].pctStart ? 'bg-success/40' : 'bg-ops-border'}`} />
              )}
            </React.Fragment>
          )
        })}
      </div>

      {/* Progress bar */}
      <div className="flex-1 h-0.5 bg-ops-border rounded overflow-hidden max-w-36">
        <div
          className={`h-full rounded transition-all duration-500
                      ${isError ? 'bg-danger' : isDone ? 'bg-success' : 'bg-amber-400'}`}
          style={{ width: `${pct}%` }}
        />
      </div>

      {/* Status message */}
      <div className="flex-1 text-[11px] text-ops-muted truncate">
        {status?.message ?? (isConnected ? 'Ready' : 'Offline')}
      </div>

      {/* Live dot */}
      <div className="flex items-center gap-1 shrink-0">
        <div className={`w-1.5 h-1.5 rounded-full ${isConnected ? 'bg-success' : 'bg-ops-border'}`} />
        <span className="text-[10px] text-ops-muted hidden sm:block">{isConnected ? 'Live' : 'Offline'}</span>
      </div>
    </div>
  )
}

// ── Export ─────────────────────────────────────────────────────────────────
export default function PipelineStatusBar({ status, isConnected, expanded = false }: Props) {
  if (expanded) return <ExpandedPipeline status={status} isConnected={isConnected} />
  return <CompactBar status={status} isConnected={isConnected} />
}
