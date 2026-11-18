import { useMemo, useState } from 'react'
import type { Lead } from '@/types/api'
import { concentration, fixed, int } from '@/lib/format'
import { SectionTitle } from './primitives'

interface Props {
  leads: Lead[]
  activeLead: Lead | null
  onSelectLead: (lead: Lead) => void
  pdbId: string
}

export default function ScreeningTable({ leads, activeLead, onSelectLead, pdbId }: Props) {
  const [maxDeltaG, setMaxDeltaG] = useState<number>(-4.0)
  const [maxMw, setMaxMw] = useState<number>(600)
  const [maxLogp, setMaxLogp] = useState<number>(6.0)
  const [maxTpsa, setMaxTpsa] = useState<number>(160)
  const [requireHinge, setRequireHinge] = useState<boolean>(false)
  const [selectedScaffolds, setSelectedScaffolds] = useState<Set<string>>(new Set())
  const [sortBy, setSortBy] = useState<'rank' | 'delta_g' | 'pic50' | 'ensemble' | 'mw' | 'logp'>('rank')

  // Extract distinct Murcko scaffolds across the returned leads
  const scaffolds = useMemo(() => {
    const counts = new Map<string, number>()
    for (const l of leads) {
      const s = l.scaffold || l.smiles
      counts.set(s, (counts.get(s) ?? 0) + 1)
    }
    return Array.from(counts.entries()).map(([scaf, count], idx) => ({
      scaf,
      label: `Scaffold S${idx + 1}`,
      count,
    }))
  }, [leads])

  const toggleScaffold = (scaf: string) => {
    setSelectedScaffolds((prev) => {
      const next = new Set(prev)
      if (next.has(scaf)) next.delete(scaf)
      else next.add(scaf)
      return next
    })
  }

  const filteredLeads = useMemo(() => {
    const out = leads.filter((l) => {
      if (l.docking_score_kcal != null && l.docking_score_kcal > maxDeltaG) return false
      if (l.mw > maxMw) return false
      if (l.logp > maxLogp) return false
      if (l.tpsa > maxTpsa) return false
      if (requireHinge && !l.hinge_hbond) return false
      if (selectedScaffolds.size > 0 && !selectedScaffolds.has(l.scaffold || l.smiles)) return false
      return true
    })
    return [...out].sort((a, b) => {
      if (sortBy === 'delta_g') return (a.docking_score_kcal ?? 0) - (b.docking_score_kcal ?? 0)
      if (sortBy === 'pic50') return b.pic50 - a.pic50
      if (sortBy === 'ensemble') return (b.ensemble_pic50 ?? b.pic50) - (a.ensemble_pic50 ?? a.pic50)
      if (sortBy === 'mw') return a.mw - b.mw
      if (sortBy === 'logp') return a.logp - b.logp
      return a.rank - b.rank
    })
  }, [leads, maxDeltaG, maxMw, maxLogp, maxTpsa, requireHinge, selectedScaffolds, sortBy])

  if (leads.length === 0) return null

  const scaffoldLabelMap = new Map(scaffolds.map((s) => [s.scaf, s.label]))

  return (
    <section className="surface p-6 sm:p-7">
      <SectionTitle
        icon={
          <svg viewBox="0 0 20 20" className="h-4 w-4" fill="none" stroke="currentColor" strokeWidth="1.6">
            <rect x="2.5" y="3.5" width="15" height="13" rx="2" />
            <path d="M2.5 8h15M8 8v8.5M13 8v8.5" />
          </svg>
        }
        aside={
          <span className="numeral">
            Showing {int(filteredLeads.length)} of {int(leads.length)} leads &middot; Target PDB {pdbId}
          </span>
        }
      >
        Affinity Ranking, Physicochemical &amp; Scaffold Filter Table
      </SectionTitle>

      {/* Interactive Slider Filters & Scaffold Checkboxes */}
      <div className="mt-3 grid gap-4 rounded-panel border border-rule bg-sunken p-4 lg:grid-cols-12">
        <div className="grid gap-3 sm:grid-cols-2 lg:col-span-8 lg:grid-cols-4">
          <div>
            <div className="flex justify-between text-micro">
              <span className="label">Max &Delta;G (kcal/mol)</span>
              <span className="numeral font-medium text-ink">&le; {fixed(maxDeltaG, 1)}</span>
            </div>
            <input
              type="range"
              min={-11.5}
              max={-4.0}
              step={0.2}
              value={maxDeltaG}
              onChange={(e) => setMaxDeltaG(Number(e.target.value))}
              className="mt-1.5 w-full accent-accent"
            />
          </div>

          <div>
            <div className="flex justify-between text-micro">
              <span className="label">Max MW (Da)</span>
              <span className="numeral font-medium text-ink">&le; {maxMw}</span>
            </div>
            <input
              type="range"
              min={250}
              max={600}
              step={10}
              value={maxMw}
              onChange={(e) => setMaxMw(Number(e.target.value))}
              className="mt-1.5 w-full accent-accent"
            />
          </div>

          <div>
            <div className="flex justify-between text-micro">
              <span className="label">Max LogP</span>
              <span className="numeral font-medium text-ink">&le; {fixed(maxLogp, 1)}</span>
            </div>
            <input
              type="range"
              min={1.0}
              max={6.0}
              step={0.2}
              value={maxLogp}
              onChange={(e) => setMaxLogp(Number(e.target.value))}
              className="mt-1.5 w-full accent-accent"
            />
          </div>

          <div>
            <div className="flex justify-between text-micro">
              <span className="label">Max TPSA (&Aring;&sup2;)</span>
              <span className="numeral font-medium text-ink">&le; {maxTpsa}</span>
            </div>
            <input
              type="range"
              min={40}
              max={160}
              step={5}
              value={maxTpsa}
              onChange={(e) => setMaxTpsa(Number(e.target.value))}
              className="mt-1.5 w-full accent-accent"
            />
          </div>
        </div>

        <div className="flex flex-wrap items-center justify-between gap-3 border-t border-rule pt-3 lg:col-span-4 lg:border-l lg:border-t-0 lg:pl-4 lg:pt-0">
          <label className="inline-flex cursor-pointer items-center gap-2 text-2xs text-ink-2">
            <input
              type="checkbox"
              checked={requireHinge}
              onChange={(e) => setRequireHinge(e.target.checked)}
              className="h-4 w-4 accent-accent"
            />
            <span>Require Met793 Hinge H-bond</span>
          </label>

          <div className="flex items-center gap-1.5">
            <span className="label">Sort:</span>
            <select
              value={sortBy}
              onChange={(e) => setSortBy(e.target.value as any)}
              aria-label="Sort candidates by"
              className="rounded-md border border-rule-strong bg-panel px-2 py-1 text-micro text-ink"
            >
              <option value="rank">Pareto Rank</option>
              <option value="delta_g">Docking &Delta;G (kcal/mol)</option>
              <option value="ensemble">Ensemble pIC50 (XGB+KRR+ChemBERTa)</option>
              <option value="pic50">XGBoost pIC50</option>
              <option value="mw">Molecular Weight</option>
              <option value="logp">LogP</option>
            </select>
          </div>
        </div>

        {/* Scaffold Isolation Checkboxes */}
        <div className="flex flex-wrap items-center gap-2 border-t border-rule pt-3 lg:col-span-12">
          <span className="label mr-1">Isolate Scaffolds:</span>
          {scaffolds.map((s) => {
            const checked = selectedScaffolds.has(s.scaf)
            return (
              <label
                key={s.scaf}
                className={`inline-flex cursor-pointer items-center gap-1.5 rounded-full border px-2.5 py-1 text-micro transition-colors ${
                  checked
                    ? 'border-accent bg-accent-soft text-accent font-medium'
                    : 'border-rule bg-panel text-ink-2 hover:border-rule-strong'
                }`}
                title={`Bemis-Murcko scaffold: ${s.scaf}`}
              >
                <input
                  type="checkbox"
                  checked={checked}
                  onChange={() => toggleScaffold(s.scaf)}
                  className="h-3 w-3 accent-accent"
                />
                <span>{s.label}</span>
                <span className="numeral text-ink-3">({s.count})</span>
              </label>
            )
          })}
          {selectedScaffolds.size > 0 && (
            <button
              type="button"
              onClick={() => setSelectedScaffolds(new Set())}
              className="text-micro text-accent hover:underline"
            >
              Clear scaffold filter
            </button>
          )}
        </div>
      </div>

      {/* Interactive Data Table */}
      <div className="mt-4 overflow-x-auto rounded-panel border border-rule">
        <table className="w-full border-collapse text-left text-2xs">
          <thead>
            <tr className="border-b border-rule bg-sunken text-micro uppercase tracking-wider text-ink-3">
              <th className="px-3.5 py-2.5">Rank</th>
              <th className="px-3.5 py-2.5">Candidate</th>
              <th className="px-3.5 py-2.5">Scaffold</th>
              <th className="px-3.5 py-2.5 text-right">&Delta;G (kcal/mol)</th>
              <th className="px-3.5 py-2.5">Pocket Contacts</th>
              <th className="px-3.5 py-2.5 text-right">IC50 (QSAR)</th>
              <th className="px-3.5 py-2.5 text-right">Ensemble pIC50</th>
              <th className="px-3.5 py-2.5 text-right">MW</th>
              <th className="px-3.5 py-2.5 text-right">LogP</th>
              <th className="px-3.5 py-2.5 text-right">TPSA</th>
              <th className="px-3.5 py-2.5 text-right">HBD/HBA</th>
              <th className="px-3.5 py-2.5 text-right">Action</th>
            </tr>
          </thead>
          <tbody className="divide-y divide-rule bg-panel">
            {filteredLeads.map((l) => {
              const isSelected = activeLead?.molecule_id === l.molecule_id
              const ic50 = concentration(l.ic50_nm)
              const scafTag = scaffoldLabelMap.get(l.scaffold || l.smiles) ?? 'S1'
              return (
                <tr
                  key={l.molecule_id}
                  onClick={() => {
                    onSelectLead(l)
                    document.getElementById('docking-workspace')?.scrollIntoView({ behavior: 'smooth' })
                  }}
                  className={`cursor-pointer transition-colors ${
                    isSelected ? 'bg-accent-soft/45' : 'hover:bg-sunken/70'
                  }`}
                >
                  <td className="numeral px-3.5 py-2.5 font-semibold text-accent">#{l.rank}</td>
                  <td className="px-3.5 py-2.5">
                    <div className="numeral font-medium text-ink">{l.molecule_id}</div>
                    {l.name && <div className="truncate max-w-[160px] text-micro text-ink-3">{l.name}</div>}
                  </td>
                  <td className="whitespace-nowrap px-3.5 py-2.5">
                    <span
                      className="badge whitespace-nowrap border border-rule bg-sunken text-ink-2"
                      title={l.scaffold}
                    >
                      {scafTag}
                    </span>
                  </td>
                  <td className="numeral whitespace-nowrap px-3.5 py-2.5 text-right font-semibold text-good">
                    {l.docking_score_kcal != null ? `${fixed(l.docking_score_kcal, 2)}` : '—'}
                  </td>
                  <td className="px-3.5 py-2.5">
                    <div className="flex flex-wrap items-center gap-1">
                      {l.hinge_hbond ? (
                        <span className="badge whitespace-nowrap bg-warn-soft text-warn">Met793 H-Bond</span>
                      ) : (
                        <span className="badge whitespace-nowrap border border-rule bg-sunken text-ink-3">Non-hinge</span>
                      )}
                      {l.cys797_distance_a != null && (
                        <span className="numeral badge whitespace-nowrap bg-sunken text-ink-2">
                          Cys797 {fixed(l.cys797_distance_a, 1)}&Aring;
                        </span>
                      )}
                      {l.gatekeeper_clash && (
                        <span className="badge whitespace-nowrap bg-crit-soft text-crit">790 Clash</span>
                      )}
                    </div>
                  </td>
                  <td className="numeral whitespace-nowrap px-3.5 py-2.5 text-right text-ink">
                    {ic50.value} {ic50.unit}
                  </td>
                  <td className="numeral whitespace-nowrap px-3.5 py-2.5 text-right font-medium text-accent">
                    {fixed(l.ensemble_pic50 ?? l.pic50, 2)}
                    <span className="ml-1 text-micro text-ink-3">(XGB {fixed(l.pic50, 2)})</span>
                  </td>
                  <td className="numeral whitespace-nowrap px-3.5 py-2.5 text-right text-ink-2">{fixed(l.mw, 0)}</td>
                  <td className="numeral whitespace-nowrap px-3.5 py-2.5 text-right text-ink-2">{fixed(l.logp, 1)}</td>
                  <td className="numeral whitespace-nowrap px-3.5 py-2.5 text-right text-ink-2">{fixed(l.tpsa, 0)}</td>
                  <td className="numeral whitespace-nowrap px-3.5 py-2.5 text-right text-ink-2">
                    {l.hbd}/{l.hba}
                  </td>
                  <td className="whitespace-nowrap px-3.5 py-2.5 text-right">
                    <button
                      type="button"
                      onClick={(e) => {
                        e.stopPropagation()
                        onSelectLead(l)
                        document.getElementById('docking-workspace')?.scrollIntoView({ behavior: 'smooth' })
                      }}
                      className={`btn-quiet whitespace-nowrap px-2.5 py-1 text-micro ${
                        isSelected ? 'border-accent bg-accent text-accent-on' : ''
                      }`}
                    >
                      {isSelected ? 'Viewing 3D' : 'Inspect 3D'}
                    </button>
                  </td>
                </tr>
              )
            })}
          </tbody>
        </table>
      </div>
    </section>
  )
}
