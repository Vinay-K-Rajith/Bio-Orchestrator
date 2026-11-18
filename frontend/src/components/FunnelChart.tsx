import { useState } from 'react'
import type { DroppedCounts, FunnelStage, Timings } from '@/types/api'
import type { StageRecord } from '@/hooks/useScreening'
import { duration, int, pct } from '@/lib/format'
import { useCountUp } from '@/lib/motion'
import { IconFunnel } from './icons'
import { SectionTitle } from './primitives'

/**
 * The screening funnel: pool → potent → safe → leads.
 *
 * Colour is ordinal, not categorical. The four stages are one hue stepped by
 * lightness, so the ramp itself says "further down the funnel" — a four-colour
 * rainbow would spend the identity channel on an order the labels already give.
 *
 * Bar lengths are logarithmic and say so. On a linear scale the last stage (5
 * out of 10,000) is a sub-pixel sliver, which hides precisely the part being
 * asked about; log keeps every stage readable while still showing that each
 * one is a large reduction.
 */

interface StageSpec {
  key: FunnelStage
  label: string
  description: string
  fill: string
  text: string
}

const STAGES: readonly StageSpec[] = [
  {
    key: 'pool',
    label: 'Candidate pool',
    description: 'Every compound scored at startup',
    fill: 'bg-stage-1',
    text: 'text-ink-2',
  },
  {
    key: 'potent',
    label: 'Predicted binders',
    description: 'Cleared the potency cutoff',
    fill: 'bg-stage-2',
    text: 'text-ink-2',
  },
  {
    key: 'safe',
    label: 'Safe & drug-like',
    description: 'Passed alerts, Lipinski and the applicability domain',
    fill: 'bg-stage-3',
    text: 'text-ink-2',
  },
  {
    key: 'leads',
    label: 'Elite leads',
    description: 'Pareto-ranked and returned',
    fill: 'bg-stage-4',
    text: 'text-accent',
  },
]

/** log1p compresses a 10,000:5 ratio into a range the eye can still compare. */
function barWidth(value: number, max: number): number {
  if (value <= 0) return 0
  return Math.max(4, (Math.log1p(value) / Math.log1p(Math.max(max, 1))) * 100)
}

function Bar({
  spec,
  value,
  previous,
  max,
  active,
  record,
}: {
  spec: StageSpec
  value: number | undefined
  previous: number | undefined
  max: number
  active: boolean
  record: StageRecord | undefined
}) {
  const [hovered, setHovered] = useState(false)
  const known = value !== undefined
  const animated = useCountUp(known ? value : 0, 800)
  const retained = known && previous ? value / previous : null

  return (
    <div
      className="group relative"
      onMouseEnter={() => setHovered(true)}
      onMouseLeave={() => setHovered(false)}
      onFocus={() => setHovered(true)}
      onBlur={() => setHovered(false)}
      tabIndex={known ? 0 : -1}
      role="figure"
      aria-label={`${spec.label}: ${known ? int(value) : 'pending'}`}
    >
      <div className="mb-1.5 flex items-baseline justify-between gap-3">
        <span
          className={`flex items-center gap-1.5 text-xs ${known ? spec.text : 'text-ink-3'}`}
        >
          {active && <span className="h-1.5 w-1.5 animate-pulse-dot rounded-full bg-accent" />}
          {spec.label}
        </span>
        <span className="flex items-baseline gap-2.5">
          {retained !== null && (
            <span className="numeral text-micro text-ink-3">{pct(retained, 1)} kept</span>
          )}
          <span className={`numeral text-sm font-medium ${known ? 'text-ink' : 'text-ink-3'}`}>
            {known ? int(animated) : '—'}
          </span>
        </span>
      </div>

      <div className="track h-2">
        <div
          className={`h-full rounded-full ${spec.fill} transition-[width,opacity] duration-700 ease-settle`}
          style={{ width: known ? `${barWidth(value, max)}%` : '0%', opacity: known ? 1 : 0 }}
        />
      </div>

      {/* Per-mark detail on hover: the exact count, the stage's own message
          and what it cost, rather than a number floating without provenance. */}
      {hovered && known && (
        <div
          role="tooltip"
          className="pointer-events-none absolute left-0 top-full z-20 mt-2 w-full animate-fade-in
                     rounded-panel border border-rule-strong bg-panel p-3 shadow-lift"
        >
          <div className="flex items-baseline justify-between gap-3">
            <span className="text-2xs font-medium text-ink">{spec.label}</span>
            <span className="numeral text-2xs text-ink-2">{int(value)} compounds</span>
          </div>
          <p className="mt-1 text-micro leading-relaxed text-ink-3">
            {record?.message ?? spec.description}
          </p>
          {record?.seconds != null && (
            <p className="numeral mt-1 text-micro text-ink-3">
              stage completed in {duration(record.seconds)}
            </p>
          )}
        </div>
      )}
    </div>
  )
}

interface Props {
  counts: Partial<Record<FunnelStage, number>>
  dropped: DroppedCounts
  timings: Timings | undefined
  activeStage: FunnelStage | null
  log: StageRecord[]
}

export default function FunnelChart({ counts, dropped, timings, activeStage, log }: Props) {
  const max = counts.pool ?? 10_000
  const attrition = [
    { label: 'Lipinski', n: dropped.lipinski ?? 0, tone: 'bg-warn-soft text-warn' },
    { label: 'structural alerts', n: dropped.alerts ?? 0, tone: 'bg-crit-soft text-crit' },
    { label: 'out of domain', n: dropped.out_of_domain ?? 0, tone: 'bg-info-soft text-info' },
  ].filter((d) => d.n > 0)

  const totalStageTime =
    timings && (timings.stage1 ?? 0) + (timings.stage2 ?? 0) + (timings.stage3 ?? 0)

  return (
    <section className="surface flex h-full flex-col p-6">
      <SectionTitle
        icon={<IconFunnel />}
        aside={
          totalStageTime ? (
            <span className="numeral">{duration(totalStageTime)} of compute</span>
          ) : undefined
        }
      >
        Funnel
      </SectionTitle>

      <div className="space-y-4">
        {STAGES.map((spec, i) => {
          const prevSpec = STAGES[i - 1]
          return (
            <Bar
              key={spec.key}
              spec={spec}
              value={counts[spec.key]}
              previous={prevSpec ? counts[prevSpec.key] : undefined}
              max={max}
              active={activeStage === spec.key}
              record={log.find((r) => r.stage === spec.key)}
            />
          )
        })}
      </div>

      {attrition.length > 0 && (
        <div className="mt-6 border-t border-rule pt-4">
          <div className="label mb-2">Stage 2 attrition</div>
          <div className="flex flex-wrap gap-1.5">
            {attrition.map((d) => (
              <span key={d.label} className={`badge ${d.tone}`}>
                <span className="numeral">{int(d.n)}</span> {d.label}
              </span>
            ))}
          </div>
        </div>
      )}

      <p className="mt-auto pt-5 text-micro leading-relaxed text-ink-3">
        Bar lengths are logarithmic — a linear scale would render the final
        stages as invisible slivers. Read the numbers, not the widths, for ratios.
      </p>
    </section>
  )
}
