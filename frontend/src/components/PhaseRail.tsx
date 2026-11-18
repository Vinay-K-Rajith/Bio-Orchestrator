import type { ScreenPhase } from '@/types/api'

/**
 * A three-step rail under the masthead: parse, screen, write.
 *
 * It advances only when the backend actually emits the corresponding event, so
 * it is a report of where the run is rather than a timer pretending to be one.
 * A stage that takes a while simply stays lit instead of creeping forward.
 */

const STEPS = [
  { key: 'parsing', label: 'Parsing brief' },
  { key: 'screening', label: 'Screening pool' },
  { key: 'reporting', label: 'Writing summary' },
] as const

const ORDER: Record<ScreenPhase, number> = {
  idle: -1,
  parsing: 0,
  screening: 1,
  reporting: 2,
  done: 3,
  error: -1,
}

export default function PhaseRail({ phase }: { phase: ScreenPhase }) {
  const active = phase === 'parsing' || phase === 'screening' || phase === 'reporting'
  const position = ORDER[phase] ?? -1

  return (
    <div
      className="overflow-hidden transition-[height,opacity] duration-500 ease-settle"
      style={{ height: active ? 34 : 0, opacity: active ? 1 : 0 }}
      aria-hidden={!active}
    >
      <div className="mx-auto flex max-w-[84rem] items-center gap-4 px-6 py-2 sm:px-8" role="status">
        {STEPS.map((step, i) => {
          const done = position > i
          const current = position === i
          return (
            <div key={step.key} className="flex flex-1 items-center gap-2.5">
              <span
                className={`h-1.5 w-1.5 shrink-0 rounded-full transition-colors duration-300 ${
                  current ? 'animate-pulse-dot bg-accent' : done ? 'bg-good' : 'bg-rule-strong'
                }`}
              />
              <span
                className={`hidden shrink-0 text-micro transition-colors duration-300 sm:inline ${
                  current ? 'text-ink' : done ? 'text-ink-3' : 'text-ink-3'
                }`}
              >
                {step.label}
              </span>
              <div className="track h-[3px] flex-1">
                <div
                  className={`relative h-full rounded-full transition-[width] duration-700 ease-settle ${
                    done ? 'bg-good/60' : 'bg-accent'
                  }`}
                  style={{ width: done ? '100%' : current ? '55%' : '0%' }}
                >
                  {current && (
                    <span className="absolute inset-y-0 right-0 w-6 animate-sweep rounded-full bg-accent" />
                  )}
                </div>
              </div>
            </div>
          )
        })}
      </div>
    </div>
  )
}
