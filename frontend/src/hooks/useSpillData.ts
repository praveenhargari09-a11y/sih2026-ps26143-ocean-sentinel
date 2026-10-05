import { useState, useEffect, useCallback } from 'react'
import type { SpillRecord, DriftResult, VesselInfo } from '../types'
import {
  listSpills, getSpillDetail, getVessels, getAISTrack,
  detectSpill, generateSyntheticDemo, detectIncident, runDrift as apiRunDrift,
} from '../api/client'

interface SpillDataActions {
  selectSpill:    (spill: SpillRecord) => void
  selectVessel:   (mmsi: string | null) => void
  runDetection:   (formData: FormData) => Promise<SpillRecord | null>
  runDrift:       () => Promise<void>
  loadVessels:    () => Promise<void>
  generateDemo:   () => Promise<void>
  loadIncident:   (incidentId: string) => Promise<void>
  clearError:     () => void
}

interface IncidentMetadata {
  data_source?: string;
  ais_source?: string;
  sar_source?: string;
  advection_model?: string;
  ais_status?: string;
}

interface UseSpillDataReturn {
  spills:             SpillRecord[]
  selectedSpill:      SpillRecord | null
  driftResult:        DriftResult | null
  vessels:            VesselInfo[]
  selectedVesselMMSI: string | null
  isLoading:          boolean
  error:              string | null
  incidentMetadata:   IncidentMetadata
  actions:            SpillDataActions
}

export function useSpillData(): UseSpillDataReturn {
  const [spills,             setSpills]             = useState<SpillRecord[]>([])
  const [selectedSpill,      setSelectedSpill]      = useState<SpillRecord | null>(null)
  const [driftResult,        setDriftResult]        = useState<DriftResult | null>(null)
  const [vessels,            setVessels]            = useState<VesselInfo[]>([])
  const [selectedVesselMMSI, setSelectedVesselMMSI] = useState<string | null>(null)
  const [isLoading,          setIsLoading]          = useState(false)
  const [error,              setError]              = useState<string | null>(null)
  const [incidentMetadata,   setIncidentMetadata]   = useState<IncidentMetadata>({})

  // Load initial demo data on mount
  useEffect(() => {
    ;(async () => {
      try {
        setIsLoading(true)
        const resp = await generateSyntheticDemo()
        const spill = resp.spill as SpillRecord
        setSpills([spill])
        setSelectedSpill(spill)
        if (resp.drift) setDriftResult(resp.drift as unknown as DriftResult)
        if (resp.vessels?.length) setVessels(resp.vessels as unknown as VesselInfo[])
        setIncidentMetadata({
          data_source: resp.data_source,
          ais_source: resp.ais_source,
          sar_source: resp.sar_source,
          advection_model: resp.advection_model,
          ais_status: resp.ais_status,
        })
      } catch (err) {
        console.warn('[useSpillData] Initial load failed:', err)
      } finally {
        setIsLoading(false)
      }
    })()
  }, [])

  const selectSpill = useCallback((spill: SpillRecord) => {
    setSelectedSpill(spill)
    setDriftResult(null)
    setVessels([])
    setSelectedVesselMMSI(null)
    setIncidentMetadata({})
    setError(null)
  }, [])

  const selectVessel = useCallback((mmsi: string | null) => {
    setSelectedVesselMMSI(mmsi)
  }, [])

  const runDetection = useCallback(async (formData: FormData): Promise<SpillRecord | null> => {
    try {
      setIsLoading(true); setError(null)
      const resp = await detectSpill(formData)
      // The detect endpoint returns the same bundle as demo (spill + drift + vessels)
      const spill = (resp as any).spill ?? resp
      setSpills(prev => [spill, ...prev])
      setSelectedSpill(spill)
      if ((resp as any).drift) setDriftResult((resp as any).drift as unknown as DriftResult)
      if ((resp as any).vessels?.length) setVessels((resp as any).vessels as unknown as VesselInfo[])
      setIncidentMetadata({
        data_source: (resp as any).data_source,
        ais_source: (resp as any).ais_source,
        sar_source: (resp as any).sar_source,
        advection_model: (resp as any).advection_model,
        ais_status: (resp as any).ais_status,
      })
      setSelectedVesselMMSI(null)
      return spill
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Detection failed')
      return null
    } finally { setIsLoading(false) }
  }, [])

  const runDrift = useCallback(async () => {
    if (!selectedSpill) { setError('No spill selected'); return }
    try {
      setIsLoading(true); setError(null)
      const result = await apiRunDrift(selectedSpill.id)
      setDriftResult(result as unknown as DriftResult)
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Drift analysis failed')
    } finally { setIsLoading(false) }
  }, [selectedSpill])

  const loadVessels = useCallback(async () => {
    if (!selectedSpill) { setError('No spill selected'); return }
    try {
      setIsLoading(true); setError(null)
      // Load vessel scores
      const scored = await getVessels(selectedSpill.id)
      // Load AIS tracks for all vessels in parallel
      const enriched = await Promise.all(
        scored.map(async (v) => {
          try {
            const track = await getAISTrack(v.mmsi)
            return { ...v, track_geojson: track.track_geojson }
          } catch {
            return v
          }
        })
      )
      setVessels(enriched as VesselInfo[])
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Vessel attribution failed')
    } finally { setIsLoading(false) }
  }, [selectedSpill])

  const generateDemo = useCallback(async () => {
    try {
      setIsLoading(true); setError(null)
      const resp = await generateSyntheticDemo()

      const spill = resp.spill as SpillRecord
      setSpills(prev => prev.some(s => s.id === spill.id) ? prev : [spill, ...prev])
      setSelectedSpill(spill)
      if (resp.drift) setDriftResult(resp.drift as unknown as DriftResult)
      if (resp.vessels?.length) setVessels(resp.vessels as unknown as VesselInfo[])
      setIncidentMetadata({
        data_source: resp.data_source,
        ais_source: resp.ais_source,
        sar_source: resp.sar_source,
        advection_model: resp.advection_model,
        ais_status: resp.ais_status,
      })
      setSelectedVesselMMSI(null)
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Demo generation failed')
    } finally { setIsLoading(false) }
  }, [])

  const loadIncident = useCallback(async (incidentId: string) => {
    try {
      setIsLoading(true); setError(null)
      
      // Generate a spill_id locally so we can connect to the WS *during* the detection
      const generatedSpillId = Math.random().toString(36).substring(2, 10)
      
      const resp = await detectIncident(incidentId, generatedSpillId)
      const spill = resp.spill as SpillRecord
      setSpills(prev => prev.some(s => s.id === spill.id) ? prev : [spill, ...prev])
      setSelectedSpill(spill)
      if (resp.drift) setDriftResult(resp.drift as unknown as DriftResult)
      if (resp.vessels?.length) setVessels(resp.vessels as unknown as VesselInfo[])
      setIncidentMetadata({
        data_source: resp.data_source,
        ais_source: resp.ais_source,
        sar_source: resp.sar_source,
        advection_model: resp.advection_model,
        ais_status: resp.ais_status,
      })
      setSelectedVesselMMSI(null)
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Incident detection failed')
    } finally { setIsLoading(false) }
  }, [])

  const clearError = useCallback(() => setError(null), [])

  return {
    spills, selectedSpill, driftResult, vessels,
    selectedVesselMMSI, isLoading, error, incidentMetadata,
    actions: { selectSpill, selectVessel, runDetection, runDrift, loadVessels, generateDemo, loadIncident, clearError },
  }
}
