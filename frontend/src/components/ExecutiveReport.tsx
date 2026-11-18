import { useState } from 'react'
import type { Citation, ModelSummary, ParetoSummary, ScreeningParams } from '@/types/api'
import { fixed, int } from '@/lib/format'
import { IconReport } from './icons'
import { Disclosure, SectionTitle } from './primitives'

/**
 * The written summary, and the model card one click behind it.
 *
 * The card is reachable from the same screen as the predictions rather than
 * from a documentation page, because the honest accuracy figures and the known
 * limitations are only useful while the numbers they qualify are still on
 * screen.
 */

/**
 * Render [S1] / [S2, S3] markers as links into the source list.
 *
 * Only markers that resolve to a supplied citation become links; the backend
 * strips unresolvable ones, so anything left over that still fails to match is
 * rendered as plain text rather than a link that goes nowhere.
 */
function CitedProse({ text, citations }: { text: string; citations: Citation[] }) {
  if (citations.length === 0) return <>{text}</>

  const byId = new Map(citations.map((c) => [c.id, c]))
  const parts = text.split(/(\[S\d+(?:\s*,\s*S\d+)*\])/g)

  return (
    <>
      {parts.map((part, i) => {
        const match = /^\[(S\d+(?:\s*,\s*S\d+)*)\]$/.exec(part)
        if (!match) return <span key={i}>{part}</span>

        const ids = (match[1] ?? '').split(',').map((id) => id.trim())
        const known = ids.filter((id) => byId.has(id))
        if (known.length === 0) return <span key={i}>{part}</span>

        return (
          <sup key={i} className="ml-0.5 whitespace-nowrap">
            {known.map((id, j) => (
              <span key={id}>
                {j > 0 && <span className="text-ink-3">,</span>}
                <a
                  href={`#source-${id}`}
                  title={byId.get(id)?.title}
                  className="numeral rounded px-0.5 text-micro font-medium text-accent transition-colors hover:bg-accent-soft hover:text-accent-hover"
                >
                  {id}
                </a>
              </span>
            ))}
          </sup>
        )
      })}
    </>
  )
}

function ParamChips({ params }: { params: ScreeningParams }) {
  const weights: ReadonlyArray<[string, number]> = [
    ['affinity', params.weight_affinity],
    ['confidence', params.weight_confidence],
    ['safety', params.weight_safety],
    ['drug-likeness', params.weight_druglikeness],
  ]

  return (
    <div className="mt-5 border-t border-rule pt-4 lg:mt-0 lg:border-l lg:border-t-0 lg:pl-7 lg:pt-1">
      <div className="label mb-2.5">
        Parameters selected from the brief
        {params.source && <span className="ml-1.5 normal-case tracking-normal">via {params.source}</span>}
      </div>

      <div className="flex flex-wrap items-center gap-1.5">
        <span className="badge bg-accent-soft text-accent">
          <span className="numeral">&le; {params.max_ic50_nm}</span> nM
        </span>
        <span className="badge border border-rule bg-sunken text-ink-2">
          top <span className="numeral">{params.top_n}</span>
        </span>
        {params.require_lipinski && (
          <span className="badge border border-rule bg-sunken text-ink-2">Lipinski enforced</span>
        )}
        <span className="badge border border-rule bg-sunken text-ink-2">
          {params.max_structural_alerts === 0 ? (
            'no structural alerts'
          ) : (
            <>
              <span className="numeral">&le; {params.max_structural_alerts}</span> alerts
            </>
          )}
        </span>
        {params.require_in_domain && (
          <span className="badge bg-info-soft text-info">in-domain only</span>
        )}
      </div>

      {/* Objective weights: one measure over four subjects, so one hue and a
          direct label on each - a four-colour key would add nothing. The block
          is width-capped because these bars encode a 0-1 fraction, and a
          400px-long bar for 0.15 reads as a layout accident rather than a value. */}
      <div className="mt-3.5 grid max-w-xl gap-x-8 gap-y-1.5 sm:grid-cols-2 lg:grid-cols-1">
        {weights.map(([name, w]) => (
          <div key={name} className="flex items-center gap-2.5">
            <span className="w-20 shrink-0 text-micro text-ink-3">{name}</span>
            <div className="track h-1 flex-1">
              <div
                className="h-full rounded-full bg-accent/70 transition-[width] duration-700 ease-settle"
                style={{ width: `${w * 100}%` }}
              />
            </div>
            <span className="numeral w-8 shrink-0 text-right text-micro text-ink-2">
              {fixed(w, 2)}
            </span>
          </div>
        ))}
      </div>

      {params.rationale && (
        <p className="mt-3.5 border-l-2 border-rule pl-3 font-display text-2xs italic leading-relaxed text-ink-2">
          {params.rationale}
        </p>
      )}
    </div>
  )
}

function SplitComparison({ random, scaffold }: { random: number; scaffold: number }) {
  const max = Math.max(random, scaffold, 0.001)
  const rows: ReadonlyArray<[string, number, string]> = [
    ['Random split', random, 'text-ink-3'],
    ['Scaffold split', scaffold, 'text-ink'],
  ]

  return (
    <div className="space-y-2">
      {rows.map(([label, value, tone]) => (
        <div key={label} className="flex items-center gap-3">
          <span className={`w-24 shrink-0 text-micro ${tone}`}>{label}</span>
          <div className="track h-2 flex-1">
            <div
              className={`h-full rounded-full transition-[width] duration-700 ease-settle ${
                label === 'Scaffold split' ? 'bg-accent' : 'bg-accent/35'
              }`}
              style={{ width: `${(value / max) * 100}%` }}
            />
          </div>
          <span className={`numeral w-10 shrink-0 text-right text-2xs ${tone}`}>
            {fixed(value, 3)}
          </span>
        </div>
      ))}
    </div>
  )
}

interface Props {
  report: string
  source: string
  citations: Citation[]
  model: ModelSummary | undefined
  params: ScreeningParams | null
  pareto: ParetoSummary | undefined
}

export default function ExecutiveReport({
  report,
  source,
  citations,
  model,
  params,
  pareto,
}: Props) {
  const [open, setOpen] = useState(false)
  const gap =
    model?.random_r2 !== undefined && model.scaffold_r2 !== undefined
      ? model.random_r2 - model.scaffold_r2
      : null

  return (
    <section className="surface p-6 sm:p-7">
      <SectionTitle
        icon={<IconReport />}
        aside={
          <span className="badge border border-rule bg-sunken text-ink-3">
            {source === 'deterministic' ? 'template' : source}
          </span>
        }
      >
        Executive summary
      </SectionTitle>

      {/* Prose and the parameters that produced it, side by side on the
          golden ratio: the measure stays at a readable ~68ch, and the space a
          capped measure would leave empty holds the contract instead. */}
      <div className="grid lg:grid-cols-[1.618fr_1fr] lg:gap-8">
        <p
          className="max-w-[68ch] font-display text-[1.0625rem] leading-[1.65] text-pretty text-ink-2"
          style={{ fontVariationSettings: "'SOFT' 22, 'WONK' 0, 'opsz' 24" }}
        >
          <CitedProse text={report} citations={citations} />
        </p>

        {params && <ParamChips params={params} />}
      </div>

      <div className="mt-5 border-t border-rule pt-4">
        <button
          type="button"
          onClick={() => setOpen((v) => !v)}
          className="inline-flex items-center gap-1.5 text-2xs font-medium text-accent transition-colors hover:text-accent-hover"
          aria-expanded={open}
        >
          <svg
            viewBox="0 0 12 12"
            className="h-3 w-3 transition-transform duration-300 ease-settle"
            style={{ transform: `rotate(${open ? 90 : 0}deg)` }}
            fill="none"
            stroke="currentColor"
            strokeWidth="1.6"
            strokeLinecap="round"
            strokeLinejoin="round"
            aria-hidden="true"
          >
            <path d="M4.5 2.5 8.5 6l-4 3.5" />
          </svg>
          Model card and limitations
        </button>

        <Disclosure open={open}>
          {model && (
            <div className="mt-4 grid gap-6 lg:grid-cols-2">
              <div>
                <div className="label mb-3">Affinity model, held-out accuracy</div>
                {model.random_r2 !== undefined ? (
                  <SplitComparison random={model.random_r2} scaffold={model.scaffold_r2} />
                ) : (
                  <div className="numeral text-2xs text-ink-2">
                    Scaffold R&sup2; {fixed(model.scaffold_r2, 3)}
                  </div>
                )}

                <div className="mt-4 grid grid-cols-2 gap-4">
                  <div>
                    <div className="label">RMSE</div>
                    <div className="numeral mt-0.5 text-sm text-ink">
                      {fixed(model.scaffold_rmse, 3)}
                      <span className="ml-1 text-micro text-ink-3">log units</span>
                    </div>
                  </div>
                  <div>
                    <div className="label">Interval coverage</div>
                    <div className="numeral mt-0.5 text-sm text-ink">
                      {fixed(model.conformal_coverage, 3)}
                      <span className="ml-1 text-micro text-ink-3">vs 0.90 target</span>
                    </div>
                  </div>
                </div>

                {gap !== null && (
                  <p className="mt-4 text-2xs leading-relaxed text-pretty text-ink-2">
                    The two R&sup2; values differ by{' '}
                    <span className="numeral text-ink">{fixed(gap, 3)}</span>. A random split lets
                    near-identical analogs fall on both sides, so it overstates accuracy; the
                    scaffold figure is the one that describes performance on genuinely new
                    chemistry, and it is what every confidence score here rests on.
                  </p>
                )}
              </div>

              <div className="space-y-3.5 text-2xs leading-relaxed text-pretty text-ink-2">
                {model.herg_available && model.herg_auc != null && (
                  <p>
                    hERG cardiotoxicity classifier, ROC-AUC{' '}
                    <span className="numeral text-ink">{fixed(model.herg_auc, 3)}</span> on a
                    scaffold split. ChEMBL hERG data is publication-biased toward positives, so read
                    this as a triage signal rather than a verdict.
                  </p>
                )}

                {pareto && pareto.n > 0 && (
                  <p>
                    Ranking used <span className="numeral text-ink">{int(pareto.n_fronts)}</span>{' '}
                    Pareto fronts over <span className="numeral text-ink">{int(pareto.n)}</span>{' '}
                    candidates;{' '}
                    <span className="numeral text-ink">{int(pareto.front_sizes[0] ?? 0)}</span> were
                    non-dominated. Weights only order molecules within a front, so no amount of
                    weighting promotes a dominated compound.
                  </p>
                )}

                <p className="rounded-panel border border-warn/30 bg-warn-soft p-3 text-warn">
                  These are QSAR predictions from 2D structure, not experimental measurements. There
                  is no binding pose, no ADME and no selectivity data, and IC50 is not Kd. Use for
                  triage before docking or assay, never as evidence of activity.
                </p>
              </div>
            </div>
          )}
        </Disclosure>
      </div>
    </section>
  )
}
