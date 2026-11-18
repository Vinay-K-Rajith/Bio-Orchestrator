import { useEffect, useState } from 'react'
import type { CSSProperties, ReactNode } from 'react'
import AppHeader from '@/components/AppHeader'
import DockingWorkspace from '@/components/DockingWorkspace'
import EvidencePanel from '@/components/EvidencePanel'
import ExecutiveReport from '@/components/ExecutiveReport'
import FunnelChart from '@/components/FunnelChart'
import FunnelIllustration from '@/components/FunnelIllustration'
import { IconMolecule, IconParse, IconRank, IconScore, IconShield, LogoMark } from '@/components/icons'
import LeadCard from '@/components/LeadCard'
import PhaseRail from '@/components/PhaseRail'
import QueryConsole from '@/components/QueryConsole'
import ScreeningTable from '@/components/ScreeningTable'
import { SectionTitle } from '@/components/primitives'
import { useHealth, useScreening, useTargets } from '@/hooks/useScreening'
import { duration, int } from '@/lib/format'
import type { EgfrPdbId, Lead } from '@/types/api'

const DEFAULT_QUERY = 'Screen for safe, orally bioavailable EGFR inhibitors under 500 nM'

function Notice({ tone, title, children }: { tone: 'crit' | 'warn'; title: string; children: React.ReactNode }) {
  const styles =
    tone === 'crit' ? 'border-crit/35 bg-crit-soft text-crit' : 'border-warn/35 bg-warn-soft text-warn'
  return (
    <div className={`animate-fade-in rounded-card border p-4 ${styles}`} role="alert">
      <p className="text-sm font-medium">{title}</p>
      <div className="mt-1 text-2xs leading-relaxed text-pretty opacity-90">{children}</div>
    </div>
  )
}

function Primer() {
  const steps: ReadonlyArray<[string, string, ReactNode]> = [
    ['Parse', 'Your sentence becomes explicit screening parameters and selects the EGFR crystal target.', <IconParse />],
    ['Score', 'The 10,000-compound pool is scored with XGBoost, Tanimoto-KRR, and ChemBERTa-77M.', <IconScore />],
    ['Filter', 'Structural alerts, Lipinski and the applicability domain cut it down.', <IconShield />],
    ['Rank & Dock', 'Pareto fronts first, then 9-mode 3D conformer docking in the EGFR active cleft.', <IconRank />],
  ]

  return (
    <section className="surface animate-fade-in overflow-hidden">
      <div className="grid lg:grid-cols-[1.618fr_1fr]">
        <div className="flex flex-col justify-center p-6 sm:p-10">
          <div className="label mb-4 text-accent">How it works</div>
          <p
            className="font-display text-lg leading-[1.45] text-balance text-ink sm:text-[1.375rem]"
            style={{ fontVariationSettings: "'SOFT' 26, 'WONK' 1, 'opsz' 36" }}
          >
            Describe the compounds you want, and the funnel narrows a precomputed pool
            down to ranked leads &mdash; coupling 2D multi-modal QSAR with 3D crystallographic
            pose generation across five EGFR states (1M17, 2ITY, 4I22, 4ZAU, 5D41).
          </p>
          <p className="mt-5 max-w-[60ch] text-2xs leading-relaxed text-pretty text-ink-3">
            Every affinity carries a calibrated 90% conformal interval, a multi-modal ensemble
            score (XGBoost + Tanimoto-KRR + ChemBERTa-77M-MTR), and 9-mode 3D pocket docking
            evaluating Met793 hinge H-bonds, Thr790/Met790 gatekeeper clashes, and Cys797
            covalent proximity.
          </p>
        </div>

        <div className="flex items-center justify-center border-t border-rule bg-sunken px-6 py-8 lg:border-l lg:border-t-0">
          <div className="w-full max-w-[30rem]">
            <FunnelIllustration />
          </div>
        </div>
      </div>

      <ol className="grid gap-px border-t border-rule bg-rule sm:grid-cols-2 lg:grid-cols-4">
        {steps.map(([name, detail, icon], i) => (
          <li key={name} className="flex gap-3.5 bg-panel p-6">
            <span className="icon-tile mt-0.5">{icon}</span>
            <div className="min-w-0">
              <div className="flex items-baseline gap-2">
                <span className="numeral text-micro text-ink-3">0{i + 1}</span>
                <span className="text-sm font-medium text-ink">{name}</span>
              </div>
              <div className="mt-1 text-2xs leading-relaxed text-pretty text-ink-3">{detail}</div>
            </div>
          </li>
        ))}
      </ol>
    </section>
  )
}

export default function App() {
  const [query, setQuery] = useState(DEFAULT_QUERY)
  const [ground, setGround] = useState(true)
  const [selectedPdb, setSelectedPdb] = useState<EgfrPdbId>('1M17')
  const [activeLead, setActiveLead] = useState<Lead | null>(null)

  const { health, unreachable } = useHealth()
  const targets = useTargets()
  const screening = useScreening()
  const { phase, params, counts, dropped, log, activeStage, result, report, evidence, grounding, error, run } =
    screening

  const hasRun = phase !== 'idle'
  const leads = result?.leads ?? []

  // Sync backend-parsed PDB target into UI state
  useEffect(() => {
    const parsedPdb = params?.pdb_id?.toUpperCase() as EgfrPdbId | undefined
    if (parsedPdb && ['1M17', '2ITY', '4I22', '4ZAU', '5D41'].includes(parsedPdb)) {
      setSelectedPdb(parsedPdb)
    }
  }, [params?.pdb_id])

  // Auto-select Rank #1 lead for 3D docking inspection when screening finishes
  useEffect(() => {
    const first = result?.leads?.[0]
    if (first) {
      setActiveLead(first)
    }
  }, [result])

  return (
    <div className="min-h-screen">
      <AppHeader health={health} />
      <PhaseRail phase={phase} />

      <main className="mx-auto max-w-[84rem] space-y-6 px-6 pb-20 pt-6 sm:px-8">
        {unreachable && (
          <Notice tone="crit" title="Backend unreachable">
            Nothing responded on <code className="numeral">:8000</code>. Start it with{' '}
            <code className="numeral">uvicorn app.server:app</code> from the{' '}
            <code className="numeral">backend</code> directory, then reload.
          </Notice>
        )}

        {error && (
          <Notice tone="crit" title="The screening run failed">
            {error}
          </Notice>
        )}

        <div className="grid gap-6 lg:grid-cols-5">
          <div className="min-w-0 lg:col-span-3">
            <QueryConsole
              query={query}
              onQueryChange={setQuery}
              onRun={run}
              phase={phase}
              health={health}
              ground={ground}
              onGroundChange={setGround}
              selectedPdb={selectedPdb}
              onSelectPdb={setSelectedPdb}
            />
          </div>
          <div className="min-w-0 lg:col-span-2">
            <FunnelChart
              counts={counts}
              dropped={dropped}
              timings={result?.timings}
              activeStage={activeStage}
              log={log}
            />
          </div>
        </div>

        {!hasRun && <Primer />}

        {report && (
          <ExecutiveReport
            report={report.report}
            source={report.source}
            citations={evidence?.citations ?? []}
            model={result?.model}
            params={params}
            pareto={result?.pareto}
          />
        )}

        <EvidencePanel evidence={evidence} grounding={grounding} />

        {/* 3D EGFR Binding Pocket & Conformer Docking Studio */}
        {leads.length > 0 && (
          <DockingWorkspace
            targets={targets}
            selectedPdb={selectedPdb}
            onSelectPdb={setSelectedPdb}
            leads={leads}
            activeLead={activeLead}
            onSelectLead={setActiveLead}
          />
        )}

        {/* Interactive Affinity Ranking, Physicochemical & Scaffold Filter Table */}
        {leads.length > 0 && (
          <ScreeningTable
            leads={leads}
            activeLead={activeLead}
            onSelectLead={setActiveLead}
            pdbId={selectedPdb}
          />
        )}

        {leads.length > 0 && (
          <section>
            <SectionTitle
              icon={<IconMolecule />}
              aside={
                result ? (
                  <span className="numeral">
                    {int(leads.length)} of {int(result.funnel.safe)} survivors &middot;{' '}
                    {duration(result.total_seconds)}
                  </span>
                ) : undefined
              }
            >
              Ranked leads
            </SectionTitle>
            <p className="-mt-2 mb-5 ml-10 max-w-[64ch] text-2xs leading-relaxed text-ink-3">
              Ordered by Pareto front first, then by weighted desirability within a front.
              Click <strong>Inspect 3D Pose</strong> on any card to load its 9-mode docked conformer
              into the EGFR ({selectedPdb}) crystal pocket above.
            </p>
            <div
              className="stagger grid gap-5 md:grid-cols-2 xl:grid-cols-3"
              style={{ '--stagger-step': '70ms' } as CSSProperties}
            >
              {leads.map((lead, i) => (
                <LeadCard
                  key={`${lead.molecule_id}-${lead.rank}`}
                  lead={lead}
                  index={i}
                  isSelected={activeLead?.molecule_id === lead.molecule_id}
                  onInspect3D={setActiveLead}
                />
              ))}
            </div>
          </section>
        )}

        {result && leads.length === 0 && phase === 'done' && (
          <section className="surface animate-fade-in p-10 text-center">
            <p className="font-display text-lg text-ink">No compounds survived the funnel.</p>
            <p className="mx-auto mt-2.5 max-w-[52ch] text-2xs leading-relaxed text-pretty text-ink-2">
              {result.funnel.potent === 0 ? (
                <>
                  None of the <span className="numeral text-ink">{int(result.funnel.pool)}</span>{' '}
                  compounds cleared the potency cutoff of{' '}
                  <span className="numeral text-ink">{result.parameters.max_ic50_nm} nM</span>.
                  Ask for a looser threshold &mdash; for example &ldquo;under 1 &micro;M&rdquo;
                  &mdash; and run it again.
                </>
              ) : (
                <>
                  <span className="numeral text-ink">{int(result.funnel.potent)}</span> cleared the
                  potency cutoff, but none made it through the safety and drug-likeness filters.
                  Relax the potency threshold, or allow a structural alert, and run it again.
                </>
              )}
            </p>
          </section>
        )}

        <footer className="grid gap-6 border-t border-rule pt-6 md:grid-cols-[1fr_auto]">
          <div>
            <p className="max-w-[76ch] text-2xs leading-relaxed text-pretty text-ink-3">
              Affinities are predicted against <strong className="font-medium text-ink-2">human
              EGFR</strong> (ChEMBL target CHEMBL203) using a multi-modal ensemble (XGBoost + Tanimoto-KRR + DeepChem/ChemBERTa-77M-MTR)
              coupled with 9-mode 3D conformer docking across PDB crystal conformations 1M17, 2ITY, 4I22, 4ZAU, and 5D41.
            </p>
            <p className="mt-2.5 max-w-[76ch] text-2xs leading-relaxed text-pretty text-ink-3">
              QSAR and empirical 3D docking predictions, for research triage only. Not experimental data,
              not medical advice, and not evidence of biological activity. Every prediction carries a
              calibrated interval and an applicability-domain flag &mdash; read both before
              acting on a number.
            </p>
          </div>
          <div className="flex items-start gap-2.5 md:justify-end">
            <LogoMark className="h-6 w-6 shrink-0" />
            <div>
              <div className="font-display text-sm font-semibold leading-none text-ink">
                Bio<span className="text-accent">Orchestrator</span>
              </div>
              <div className="mt-1 text-micro text-ink-3">For research triage only</div>
            </div>
          </div>
        </footer>
      </main>
    </div>
  )
}
