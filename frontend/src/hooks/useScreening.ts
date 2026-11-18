import { useCallback, useEffect, useRef, useState } from 'react'
import type {
  DroppedCounts,
  EgfrPdbId,
  EvidenceBundle,
  FunnelStage,
  Health,
  ReportEvent,
  ScreenPhase,
  ScreenResult,
  ScreeningParams,
  StageEvent,
  TargetMetadata,
} from '@/types/api'

export interface StageRecord {
  stage: FunnelStage
  count: number
  message: string
  seconds: number | null
  at: number
}

export interface ScreeningState {
  phase: ScreenPhase
  params: ScreeningParams | null
  counts: Partial<Record<FunnelStage, number>>
  dropped: DroppedCounts
  log: StageRecord[]
  activeStage: FunnelStage | null
  result: ScreenResult | null
  report: ReportEvent | null
  evidence: EvidenceBundle | null
  grounding: boolean
  error: string | null
}

const EMPTY: ScreeningState = {
  phase: 'idle',
  params: null,
  counts: {},
  dropped: {},
  log: [],
  activeStage: null,
  result: null,
  report: null,
  evidence: null,
  grounding: false,
  error: null,
}

function parse<T>(raw: string): T | null {
  try {
    return JSON.parse(raw) as T
  } catch {
    return null
  }
}

export function useScreening() {
  const [state, setState] = useState<ScreeningState>(EMPTY)
  const sourceRef = useRef<EventSource | null>(null)

  const close = useCallback(() => {
    sourceRef.current?.close()
    sourceRef.current = null
  }, [])

  useEffect(() => close, [close])

  const run = useCallback(
    (query: string, ground = false, pdbId?: EgfrPdbId) => {
      close()
      setState({ ...EMPTY, phase: 'parsing' })

      const qs = new URLSearchParams({ query })
      if (ground) qs.set('ground', 'true')
      if (pdbId) qs.set('pdb_id', pdbId)

      const es = new EventSource(`/api/screen/stream?${qs.toString()}`)
      sourceRef.current = es

      es.addEventListener('params', (e) => {
        const params = parse<ScreeningParams>(e.data)
        if (params) setState((s) => ({ ...s, params, phase: 'screening' }))
      })

      es.addEventListener('stage', (e) => {
        const d = parse<StageEvent>(e.data)
        if (!d) return
        setState((s) => ({
          ...s,
          phase: 'screening',
          activeStage: d.stage,
          counts: { ...s.counts, [d.stage]: d.count },
          dropped: d.dropped ?? s.dropped,
          log: [
            ...s.log,
            {
              stage: d.stage,
              count: d.count,
              message: d.message,
              seconds: d.seconds ?? null,
              at: Date.now(),
            },
          ],
        }))
      })

      es.addEventListener('result', (e) => {
        const result = parse<ScreenResult>(e.data)
        if (result) setState((s) => ({ ...s, result, phase: 'reporting' }))
      })

      es.addEventListener('evidence_start', () => {
        setState((s) => ({ ...s, grounding: true }))
      })

      es.addEventListener('evidence', (e) => {
        const evidence = parse<EvidenceBundle>(e.data)
        setState((s) => ({ ...s, evidence, grounding: false }))
      })

      es.addEventListener('report', (e) => {
        const report = parse<ReportEvent>(e.data)
        if (report) setState((s) => ({ ...s, report }))
      })

      es.addEventListener('done', () => {
        setState((s) => ({ ...s, phase: 'done', activeStage: null, grounding: false }))
        close()
      })

      es.addEventListener('error', (e) => {
        const payload = (e as MessageEvent<string>).data
        const decoded = payload ? parse<{ message: string }>(payload) : null
        setState((s) => ({
          ...s,
          phase: 'error',
          activeStage: null,
          grounding: false,
          error:
            decoded?.message ??
            'Lost the screening stream. Check that uvicorn is still running on :8000.',
        }))
        close()
      })
    },
    [close],
  )

  const reset = useCallback(() => {
    close()
    setState(EMPTY)
  }, [close])

  return { ...state, run, reset }
}

export function useHealth() {
  const [health, setHealth] = useState<Health | null>(null)
  const [unreachable, setUnreachable] = useState(false)

  useEffect(() => {
    let cancelled = false
    fetch('/api/health')
      .then((r) => (r.ok ? (r.json() as Promise<Health>) : Promise.reject(new Error(r.statusText))))
      .then((h) => {
        if (!cancelled) setHealth(h)
      })
      .catch(() => {
        if (!cancelled) setUnreachable(true)
      })
    return () => {
      cancelled = true
    }
  }, [])

  return { health, unreachable }
}

export function useTargets() {
  const [targets, setTargets] = useState<TargetMetadata[]>([])

  useEffect(() => {
    let cancelled = false
    fetch('/api/targets')
      .then((r) => (r.ok ? (r.json() as Promise<{ targets: TargetMetadata[] }>) : Promise.reject()))
      .then((d) => {
        if (!cancelled && d.targets) setTargets(d.targets)
      })
      .catch(() => {})
    return () => {
      cancelled = true
    }
  }, [])

  return targets
}
