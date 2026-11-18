import { useEffect, useRef } from 'react'
import type { EgfrPdbId, Health, ScreenPhase } from '@/types/api'
import { IconBrief } from './icons'
import { SectionTitle } from './primitives'

interface Preset {
  label: string
  query: string
  pdb?: EgfrPdbId
}

const PRESETS: readonly Preset[] = [
  { label: 'Oral & safe (1M17)', query: 'Screen for safe, orally bioavailable EGFR inhibitors under 500 nM', pdb: '1M17' },
  { label: 'T790M Mutant (4I22)', query: 'Find potent EGFR T790M mutant binders under 600 nM, top 8 leads', pdb: '4I22' },
  { label: 'Covalent Cys797 (4ZAU)', query: 'Screen for covalent-pocket EGFR inhibitors in 4ZAU under 500 nM', pdb: '4ZAU' },
  { label: 'Allosteric Site (5D41)', query: 'Find novel allosteric EGFR 5D41 candidates under 800 nM', pdb: '5D41' },
  { label: 'High confidence', query: 'Give me reliable, well-characterised EGFR leads under 600 nM', pdb: '2ITY' },
]

const PDB_OPTIONS: ReadonlyArray<{ id: EgfrPdbId; short: string }> = [
  { id: '1M17', short: '1M17 · WT Active (Erlotinib)' },
  { id: '2ITY', short: '2ITY · WT (Gefitinib)' },
  { id: '4I22', short: '4I22 · T790M Mutant' },
  { id: '4ZAU', short: '4ZAU · Covalent Cys797' },
  { id: '5D41', short: '5D41 · Allosteric Site' },
]

interface Props {
  query: string
  onQueryChange: (q: string) => void
  onRun: (q: string, ground: boolean, pdbId: EgfrPdbId) => void
  phase: ScreenPhase
  health: Health | null
  ground: boolean
  onGroundChange: (v: boolean) => void
  selectedPdb: EgfrPdbId
  onSelectPdb: (pdb: EgfrPdbId) => void
}

export default function QueryConsole({
  query,
  onQueryChange,
  onRun,
  phase,
  health,
  ground,
  onGroundChange,
  selectedPdb,
  onSelectPdb,
}: Props) {
  const running = phase === 'parsing' || phase === 'screening' || phase === 'reporting'
  const areaRef = useRef<HTMLTextAreaElement>(null)

  useEffect(() => {
    const el = areaRef.current
    if (!el) return
    el.style.height = 'auto'
    el.style.height = `${Math.min(el.scrollHeight, 220)}px`
  }, [query])

  const submit = () => {
    const q = query.trim()
    if (!running && q) onRun(q, ground && tavily, selectedPdb)
  }

  const backends = health?.llm_backends
  const nebius = backends?.nebius ?? false
  const llama = backends?.llama_cpp ?? false
  const tavily = backends?.tavily ?? false
  const live = nebius || llama
  const backendName = nebius
    ? `${backends?.nebius_model?.split('/').pop() ?? 'Nemotron'} · Nebius`
    : llama
      ? 'Nemotron · llama.cpp'
      : 'Rule-based parser'

  return (
    <section className="surface p-6">
      <SectionTitle
        icon={<IconBrief />}
        aside={
          <span
            className="badge min-w-0 max-w-full border border-rule bg-sunken text-ink-2"
            title="The language layer only selects parameters and writes the summary — it never computes a prediction, so the numbers are identical whichever backend answers."
          >
            <span
              className={`h-1.5 w-1.5 rounded-full ${
                live ? 'animate-pulse-dot bg-good' : 'bg-ink-3'
              } shrink-0`}
            />
            <span className="truncate">{backendName}</span>
          </span>
        }
      >
        Screening brief
      </SectionTitle>

      <form
        onSubmit={(e) => {
          e.preventDefault()
          submit()
        }}
      >
        <textarea
          ref={areaRef}
          value={query}
          onChange={(e) => onQueryChange(e.target.value)}
          onKeyDown={(e) => {
            if (e.key === 'Enter' && (e.metaKey || e.ctrlKey)) {
              e.preventDefault()
              submit()
            }
          }}
          rows={2}
          spellCheck={false}
          placeholder="Describe the compounds you're looking for…"
          aria-label="Screening brief"
          className="field resize-none font-display text-[0.95rem] leading-relaxed"
          style={{ fontVariationSettings: "'SOFT' 20, 'WONK' 0, 'opsz' 20" }}
        />

        {/* Target EGFR Crystal Conformation Selector */}
        <div className="mt-3.5">
          <div className="label mb-1.5">Target EGFR 3D Conformation (PDB)</div>
          <div className="flex flex-wrap gap-1.5">
            {PDB_OPTIONS.map((opt) => {
              const active = selectedPdb === opt.id
              return (
                <button
                  key={opt.id}
                  type="button"
                  disabled={running}
                  onClick={() => onSelectPdb(opt.id)}
                  className={`rounded-md border px-2.5 py-1 text-micro font-medium transition-colors ${
                    active
                      ? 'border-accent bg-accent text-accent-on'
                      : 'border-rule bg-sunken text-ink-2 hover:border-rule-strong hover:text-ink'
                  }`}
                >
                  {opt.short}
                </button>
              )
            })}
          </div>
        </div>

        <label
          className={`mt-3.5 flex items-start gap-2.5 ${tavily ? 'cursor-pointer' : 'cursor-not-allowed opacity-55'}`}
          title={
            tavily
              ? undefined
              : 'Set TAVILY_API_KEY in .env to enable literature grounding.'
          }
        >
          <input
            type="checkbox"
            checked={ground && tavily}
            disabled={!tavily || running}
            onChange={(e) => onGroundChange(e.target.checked)}
            className="mt-0.5 h-4 w-4 shrink-0 accent-accent"
          />
          <span className="min-w-0">
            <span className="block text-2xs text-ink-2">Ground the summary in published literature</span>
            <span className="block text-micro leading-relaxed text-ink-3">
              {tavily
                ? 'The agent searches primary literature and regulators, reads what looks relevant, and cites it. Adds a few seconds.'
                : 'Unavailable — no Tavily key configured.'}
            </span>
          </span>
        </label>

        <div className="mt-4 flex flex-wrap items-center justify-between gap-3">
          <span className="text-2xs text-ink-3">
            <kbd className="numeral rounded border border-rule bg-sunken px-1.5 py-0.5 text-micro">
              ⌘/Ctrl
            </kbd>
            <span className="mx-1">+</span>
            <kbd className="numeral rounded border border-rule bg-sunken px-1.5 py-0.5 text-micro">
              ↵
            </kbd>
            <span className="ml-2">to run</span>
          </span>

          <button type="submit" disabled={running || !query.trim()} className="btn-primary">
            {running ? (
              <>
                <span className="relative flex h-1.5 w-8 overflow-hidden rounded-full bg-accent-on/25">
                  <span className="absolute inset-y-0 w-1/4 animate-sweep rounded-full bg-accent-on/90" />
                </span>
                Screening &amp; 3D Docking
              </>
            ) : (
              `Run screen (${selectedPdb})`
            )}
          </button>
        </div>
      </form>

      <div className="mt-4 border-t border-rule pt-3.5">
        <div className="label mb-2">Or start from</div>
        <div className="flex flex-wrap gap-2">
          {PRESETS.map((p) => (
            <button
              key={p.label}
              type="button"
              className="chip"
              disabled={running}
              title={p.query}
              onClick={() => {
                onQueryChange(p.query)
                if (p.pdb) onSelectPdb(p.pdb)
              }}
            >
              {p.label}
            </button>
          ))}
        </div>
      </div>
    </section>
  )
}
