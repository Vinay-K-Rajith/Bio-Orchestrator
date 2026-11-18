import { useState } from 'react'
import type { CSSProperties } from 'react'
import type { Lead } from '@/types/api'
import { concentration, fixed, int } from '@/lib/format'
import { Disclosure, Meter, Stat } from './primitives'
import StructureView from './StructureView'

const AXIS_MIN = 4
const AXIS_MAX = 10

const axisPos = (v: number) =>
  ((Math.min(AXIS_MAX, Math.max(AXIS_MIN, v)) - AXIS_MIN) / (AXIS_MAX - AXIS_MIN)) * 100

function IntervalPlot({ lead }: { lead: Lead }) {
  const lo = axisPos(lead.pic50_lo)
  const hi = axisPos(lead.pic50_hi)
  const point = axisPos(lead.pic50)
  const wide = lead.interval_width > 2.5

  return (
    <div className="mt-3">
      <div
        className="relative h-6"
        title={`90% prediction interval: pIC50 ${fixed(lead.pic50_lo, 2)} to ${fixed(lead.pic50_hi, 2)}`}
      >
        <div className="absolute inset-x-0 top-2.5 h-px bg-rule" />
        {[5, 6, 7, 8, 9].map((tick) => (
          <div
            key={tick}
            className="absolute top-1.5 h-1.5 w-px bg-rule-strong"
            style={{ left: `${axisPos(tick)}%` }}
          />
        ))}

        <div
          className={`absolute top-[7px] h-1.5 rounded-full transition-all duration-700 ease-settle ${
            wide ? 'bg-warn/45' : 'bg-accent/35'
          }`}
          style={{ left: `${lo}%`, width: `${Math.max(hi - lo, 1)}%` }}
        />
        <div
          className="absolute top-0.5 h-4 w-[2.5px] rounded-full bg-accent ring-2 ring-panel transition-all duration-700 ease-settle"
          style={{ left: `calc(${point}% - 1.25px)` }}
        />
      </div>
      <div className="numeral flex justify-between text-micro text-ink-3">
        <span>pIC50 4</span>
        <span className={wide ? 'text-warn' : 'text-ink-3'}>
          &plusmn;{fixed(lead.interval_width / 2, 2)} log units
        </span>
        <span>10</span>
      </div>
    </div>
  )
}

interface Props {
  lead: Lead
  index: number
  isSelected?: boolean
  onInspect3D?: (lead: Lead) => void
}

export default function LeadCard({ lead, index, isSelected, onInspect3D }: Props) {
  const [showSmiles, setShowSmiles] = useState(false)
  const ic50 = concentration(lead.ic50_nm)
  const lo = concentration(lead.ic50_nm_lo)
  const hi = concentration(lead.ic50_nm_hi)
  const d = lead.desirability
  const optimal = lead.pareto_front === 0

  return (
    <article
      className={`surface animate-rise-in relative flex flex-col overflow-hidden p-5 transition-all duration-300 hover:shadow-lift ${
        isSelected ? 'ring-2 ring-accent' : ''
      }`}
      style={{ '--i': index } as CSSProperties}
    >
      {optimal && (
        <span
          className="absolute inset-x-0 top-0 h-[3px] bg-accent"
          title="Pareto-optimal: no other survivor is better on every objective"
        />
      )}
      <header className="flex items-start justify-between gap-3">
        <div className="flex min-w-0 items-baseline gap-2.5">
          <span
            className="font-display text-2xl font-semibold leading-none text-accent"
            style={{ fontVariationSettings: "'SOFT' 30, 'WONK' 1, 'opsz' 48" }}
          >
            {lead.rank}
          </span>
          <div className="min-w-0">
            <div className="numeral truncate text-[0.8125rem] font-medium text-ink">
              {lead.molecule_id}
            </div>
            {lead.name && <div className="truncate text-micro text-ink-3">{lead.name}</div>}
          </div>
        </div>

        <span
          className={`badge shrink-0 ${
            lead.in_domain ? 'bg-good-soft text-good' : 'bg-warn-soft text-warn'
          }`}
          title={`Maximum Tanimoto similarity to any training compound: ${lead.max_similarity}`}
        >
          <svg
            viewBox="0 0 12 12"
            className="h-3 w-3"
            fill="none"
            stroke="currentColor"
            strokeWidth="1.5"
            aria-hidden="true"
          >
            {lead.in_domain ? (
              <path d="M2.5 6.3 4.8 8.6 9.5 3.9" strokeLinecap="round" strokeLinejoin="round" />
            ) : (
              <>
                <path d="M6 3.2v3.4" strokeLinecap="round" />
                <circle cx="6" cy="8.7" r="0.55" fill="currentColor" stroke="none" />
              </>
            )}
          </svg>
          {lead.in_domain ? 'In domain' : 'Extrapolation'}
          <span className="numeral opacity-70">T {fixed(lead.max_similarity, 2)}</span>
        </span>
      </header>

      <div className="mt-4">
        <StructureView smiles={lead.smiles} label={lead.molecule_id} />
      </div>

      {/* 3D Docking Summary Strip */}
      {lead.docking_score_kcal != null && (
        <div className="mt-3 flex flex-wrap items-center justify-between gap-2 rounded-panel border border-rule bg-sunken px-3 py-2 text-micro">
          <div className="flex items-center gap-2">
            <span className="label">3D Dock ({lead.pdb_id ?? '1M17'}):</span>
            <span className="numeral font-semibold text-good">
              {fixed(lead.docking_score_kcal, 2)} kcal/mol
            </span>
          </div>
          <div className="flex items-center gap-1.5">
            {lead.hinge_hbond && (
              <span className="badge bg-warn-soft text-warn">Met793 H-Bond</span>
            )}
            {onInspect3D && (
              <button
                type="button"
                onClick={() => {
                  onInspect3D(lead)
                  document.getElementById('docking-workspace')?.scrollIntoView({ behavior: 'smooth' })
                }}
                className="font-medium text-accent hover:underline"
              >
                Inspect 3D Pose &rarr;
              </button>
            )}
          </div>
        </div>
      )}

      <div className="surface-quiet mt-3 p-3.5">
        <div className="flex items-end justify-between gap-3">
          <div>
            <div className="label">Predicted IC50</div>
            <div className="numeral mt-1 flex items-baseline gap-1.5">
              <span className="text-[1.75rem] font-medium leading-none tracking-tight text-ink">
                {ic50.value}
              </span>
              <span className="text-sm text-ink-2">{ic50.unit}</span>
            </div>
          </div>
          <div className="text-right">
            <div className="label">pIC50 (XGB / Ens)</div>
            <div className="numeral mt-1 text-lg font-medium leading-none text-accent">
              {fixed(lead.pic50, 2)}
              {lead.ensemble_pic50 != null && (
                <span className="ml-1 text-xs text-ink-3">/ {fixed(lead.ensemble_pic50, 2)}</span>
              )}
            </div>
          </div>
        </div>

        <IntervalPlot lead={lead} />

        <div className="numeral mt-1.5 text-micro text-ink-3">
          90% interval {lo.value} {lo.unit} &ndash; {hi.value} {hi.unit}
        </div>
      </div>

      <div className="mt-4 grid grid-cols-4 gap-x-3 gap-y-3">
        <Stat label="MW" value={fixed(lead.mw, 0)} hint="Molecular weight (Da)" />
        <Stat label="LogP" value={fixed(lead.logp, 1)} hint="Octanol-water partition coefficient" />
        <Stat label="QED" value={fixed(lead.qed, 2)} hint="Quantitative estimate of drug-likeness, 0-1" />
        <Stat label="SA" value={fixed(lead.sa_score, 1)} hint="Synthetic accessibility, 1 easy - 10 hard" />
        <Stat label="TPSA" value={fixed(lead.tpsa, 0)} hint="Topological polar surface area (A^2)" />
        <Stat
          label="HBD / HBA"
          value={`${lead.hbd}/${lead.hba}`}
          hint="Hydrogen-bond donors / acceptors"
        />
        <Stat
          label="hERG"
          value={lead.herg_risk === null ? 'n/a' : fixed(lead.herg_risk, 2)}
          tone={lead.herg_risk !== null && lead.herg_risk > 0.5 ? 'crit' : 'default'}
          hint={
            lead.herg_risk === null
              ? 'The cardiotoxicity classifier is not loaded - this is unknown, not safe.'
              : 'Predicted probability of hERG blockade'
          }
        />
        <Stat
          label="Front"
          value={int(lead.pareto_front)}
          tone={lead.pareto_front === 0 ? 'good' : 'default'}
          hint="Pareto front index - 0 means nothing dominates it"
        />
      </div>

      <div className="mt-4 space-y-2 border-t border-rule pt-4">
        <Meter label="Affinity" value={d.affinity} />
        <Meter label="Confidence" value={d.confidence} />
        <Meter label="Safety" value={d.safety} />
        <Meter label="Drug-likeness" value={d.drug_likeness} />
      </div>

      {lead.flags.length > 0 && (
        <ul className="mt-4 space-y-1.5 border-t border-rule pt-4">
          {lead.flags.map((flag) => (
            <li key={flag} className="flex gap-2 text-micro leading-relaxed text-warn">
              <svg
                viewBox="0 0 12 12"
                className="mt-px h-3 w-3 shrink-0"
                fill="none"
                stroke="currentColor"
                strokeWidth="1.4"
                aria-hidden="true"
              >
                <path d="M6 1.6 11 10.4H1L6 1.6Z" strokeLinejoin="round" />
                <path d="M6 5v2.2" strokeLinecap="round" />
                <circle cx="6" cy="8.9" r="0.5" fill="currentColor" stroke="none" />
              </svg>
              <span>{flag}</span>
            </li>
          ))}
        </ul>
      )}

      <div className="mt-auto pt-4 flex items-center justify-between gap-2">
        <button
          type="button"
          onClick={() => setShowSmiles((v) => !v)}
          className="label transition-colors hover:text-accent"
          aria-expanded={showSmiles}
        >
          {showSmiles ? '- Hide' : '+ Show'} SMILES
        </button>
        {onInspect3D && (
          <button
            type="button"
            onClick={() => {
              onInspect3D(lead)
              document.getElementById('docking-workspace')?.scrollIntoView({ behavior: 'smooth' })
            }}
            className="btn-quiet px-2.5 py-1 text-micro"
          >
            View 3D Pose
          </button>
        )}
      </div>
      <Disclosure open={showSmiles}>
        <code className="numeral mt-2 block break-all rounded-panel bg-sunken p-2.5 text-micro leading-relaxed text-ink-2">
          {lead.smiles}
        </code>
      </Disclosure>
    </article>
  )
}
