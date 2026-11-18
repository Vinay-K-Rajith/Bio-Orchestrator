/** Presentation-only number formatting. Nothing here changes a value. */

const INT = new Intl.NumberFormat('en-US')

export const int = (n: number): string => INT.format(Math.round(n))

export const fixed = (n: number | null | undefined, digits: number): string =>
  n === null || n === undefined || Number.isNaN(n) ? '—' : n.toFixed(digits)

export const pct = (fraction: number, digits = 0): string =>
  `${(fraction * 100).toFixed(digits)}%`

/**
 * Concentrations span four orders of magnitude across a funnel, and "12000 nM"
 * is harder to read at a glance than "12.0 µM". Switch units above 1 µM and
 * keep three significant figures either side of the boundary.
 */
export function concentration(nm: number): { value: string; unit: string } {
  if (!Number.isFinite(nm)) return { value: '—', unit: '' }
  if (nm >= 1000) return { value: (nm / 1000).toFixed(nm >= 10_000 ? 0 : 1), unit: 'µM' }
  if (nm >= 10) return { value: nm.toFixed(0), unit: 'nM' }
  return { value: nm.toFixed(nm >= 1 ? 1 : 2), unit: 'nM' }
}

export const concentrationText = (nm: number): string => {
  const { value, unit } = concentration(nm)
  return unit ? `${value} ${unit}` : value
}

/** Sub-second work is the whole point of the precomputed pool; show it. */
export const duration = (seconds: number | undefined): string => {
  if (seconds === undefined) return '—'
  if (seconds < 1) return `${(seconds * 1000).toFixed(0)} ms`
  return `${seconds.toFixed(2)} s`
}
