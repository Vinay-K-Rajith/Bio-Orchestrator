// Only the count-up hook needs React state here; the rest are presentational.
import type { ReactNode } from 'react'
import { useCountUp } from '@/lib/motion'

/* ---- Meter -------------------------------------------------------------- *
 * One measure (0–1 desirability) across several subjects, so every bar takes
 * the same hue: the length is the reading, and spending the colour channel on
 * the row's identity would only re-encode the label already sitting beside it.
 * -------------------------------------------------------------------------- */

export function Meter({ label, value }: { label: string; value: number }) {
  const animated = useCountUp(value, 700)
  const pct = Math.max(0, Math.min(1, animated)) * 100

  return (
    <div className="flex items-center gap-3">
      <span className="w-[5.5rem] shrink-0 text-2xs text-ink-2">{label}</span>
      <div className="track h-[5px] flex-1">
        <div
          className="h-full rounded-full bg-accent transition-[width] duration-700 ease-settle"
          style={{ width: `${pct}%` }}
        />
      </div>
      <span className="numeral w-8 shrink-0 text-right text-2xs text-ink-2">
        {(animated * 100).toFixed(0)}
      </span>
    </div>
  )
}

/* ---- Stat --------------------------------------------------------------- */

interface StatProps {
  label: string
  value: ReactNode
  unit?: string
  tone?: 'default' | 'good' | 'warn' | 'crit' | 'info' | 'accent'
  hint?: string
}

const TONE: Record<NonNullable<StatProps['tone']>, string> = {
  default: 'text-ink',
  good: 'text-good',
  warn: 'text-warn',
  crit: 'text-crit',
  info: 'text-info',
  accent: 'text-accent',
}

export function Stat({ label, value, unit, tone = 'default', hint }: StatProps) {
  return (
    <div title={hint} className="min-w-0">
      <div className="label truncate">{label}</div>
      <div className={`numeral mt-0.5 flex items-baseline gap-1 text-sm ${TONE[tone]}`}>
        <span className="truncate font-medium">{value}</span>
        {unit && <span className="text-2xs font-normal text-ink-3">{unit}</span>}
      </div>
    </div>
  )
}

/* ---- Disclosure --------------------------------------------------------- *
 * Animated by interpolating a grid row between 0fr and 1fr rather than by
 * measuring a pixel height. Measurement has to be redone whenever the content
 * reflows - a web font landing late, a column stacking at a breakpoint - and a
 * stale measurement silently crops the last line. This can't go stale: the row
 * always resolves to whatever the content currently needs.
 * -------------------------------------------------------------------------- */

export function Disclosure({ open, children }: { open: boolean; children: ReactNode }) {
  return (
    <div
      className="grid transition-[grid-template-rows,opacity] duration-500 ease-settle"
      style={{ gridTemplateRows: open ? '1fr' : '0fr', opacity: open ? 1 : 0 }}
      aria-hidden={!open}
    >
      <div className="overflow-hidden">{children}</div>
    </div>
  )
}

/* ---- Section heading ---------------------------------------------------- */

export function SectionTitle({
  children,
  aside,
  icon,
}: {
  children: ReactNode
  aside?: ReactNode
  /** An icon from `./icons`, set in the accent tile before the heading. */
  icon?: ReactNode
}) {
  return (
    <div className="rule-title mb-5">
      {icon && <span className="icon-tile">{icon}</span>}
      <h2 className="shrink-0 whitespace-nowrap font-display text-base font-semibold tracking-[-0.01em] text-ink">
        {children}
      </h2>
      {/* The aside yields width before the heading does, so a long backend name
          truncates instead of pushing the card wider than a phone screen. */}
      {aside && (
        <span className="order-last flex min-w-0 justify-end text-2xs text-ink-3">{aside}</span>
      )}
    </div>
  )
}
