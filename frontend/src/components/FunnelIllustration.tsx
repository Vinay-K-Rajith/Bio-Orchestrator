import { useMemo } from 'react'

/**
 * The idea of the product in one picture: a crowd of candidate molecules
 * drawn left to right through three gates, leaving a short ranked list.
 *
 * It is a diagram, not data — there are no counts on it, so it can never
 * disagree with the funnel card beside it. The colours are the funnel's own
 * ordinal ramp, which means the illustration teaches the reader the colour
 * code before the first real result uses it. Positions come from a seeded
 * generator, so the picture is identical on every load.
 */

const W = 480
const H = 272

/** Pointy-top hexagon: the generic "molecule" glyph. */
function hex(cx: number, cy: number, r: number): string {
  const pts = Array.from({ length: 6 }, (_, i) => {
    const a = (Math.PI / 3) * i - Math.PI / 2
    return `${(cx + r * Math.cos(a)).toFixed(1)},${(cy + r * Math.sin(a)).toFixed(1)}`
  })
  return `M${pts.join('L')}Z`
}

function mulberry32(seed: number) {
  return () => {
    seed |= 0
    seed = (seed + 0x6d2b79f5) | 0
    let t = Math.imul(seed ^ (seed >>> 15), 1 | seed)
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296
  }
}

/** Scatter `n` points in a box with a minimum spacing, by rejection. */
function scatter(rand: () => number, n: number, box: [number, number, number, number], gap: number) {
  const [x0, y0, x1, y1] = box
  const out: Array<[number, number]> = []
  for (let tries = 0; out.length < n && tries < 4000; tries++) {
    const p: [number, number] = [x0 + rand() * (x1 - x0), y0 + rand() * (y1 - y0)]
    if (out.every(([x, y]) => Math.hypot(x - p[0], y - p[1]) >= gap)) out.push(p)
  }
  return out
}

const GATES = [
  { x: 178, label: 'potency' },
  { x: 292, label: 'safety' },
  { x: 384, label: 'Pareto' },
] as const

const COLUMNS = [
  { x: 92, label: 'Pool' },
  { x: 235, label: 'Binders' },
  { x: 338, label: 'Safe' },
  { x: 432, label: 'Leads' },
] as const

/* Upper and lower edge of the funnel envelope, sampled at each gate. */
const TOP = 'M14 26 L170 26 C 230 26, 250 62, 292 74 S 360 96, 470 98'
const BOTTOM = 'L470 186 C 360 188, 330 196, 292 204 S 230 238, 170 238 L14 238 Z'

export default function FunnelIllustration() {
  const { pool, binders, safe } = useMemo(() => {
    const rand = mulberry32(203) // CHEMBL203, for luck
    return {
      pool: scatter(rand, 46, [30, 42, 156, 222], 17),
      binders: scatter(rand, 11, [196, 82, 274, 196], 20),
      safe: scatter(rand, 4, [312, 106, 366, 176], 21),
    }
  }, [])

  const leads = [102, 142, 182]

  return (
    <svg
      viewBox={`0 0 ${W} ${H}`}
      className="h-auto w-full"
      role="img"
      aria-label="Illustration: a large pool of candidate molecules passes through potency, safety and Pareto gates, leaving three ranked leads."
    >
      {/* Envelope */}
      <path d={`${TOP} ${BOTTOM}`} className="fill-accent/[0.05] stroke-rule-strong" strokeWidth="1" />

      {/* Flow lines: the direction of travel, quietly animated. */}
      <g fill="none" strokeWidth="1.2" strokeLinecap="round" className="stroke-stage-2/50">
        {[70, 132, 194].map((y, i) => (
          <path
            key={y}
            d={`M150 ${y} C 260 ${y}, 300 ${leads[i]}, 412 ${leads[i]}`}
            strokeDasharray="2 6"
            className="animate-flow"
          />
        ))}
      </g>

      {/* Gates */}
      {GATES.map((g) => (
        <g key={g.label}>
          <line x1={g.x} x2={g.x} y1="14" y2="250" className="stroke-rule-strong" strokeDasharray="3 4" />
          <rect x={g.x - 32} y="0.5" width="64" height="20" rx="10" className="fill-panel stroke-rule-strong" />
          <text
            x={g.x}
            y="14.2"
            textAnchor="middle"
            className="fill-ink-2 font-mono"
            style={{ fontSize: 10.5 }}
          >
            {g.label}
          </text>
        </g>
      ))}

      {/* Stages, light to dark along the ramp */}
      <g strokeWidth="1.1" strokeLinejoin="round">
        {pool.map(([x, y], i) => (
          <path key={`p${i}`} d={hex(x, y, 5.2)} className="fill-stage-1/30 stroke-stage-1" />
        ))}
        {binders.map(([x, y], i) => (
          <path key={`b${i}`} d={hex(x, y, 6.2)} className="fill-stage-2/35 stroke-stage-2" />
        ))}
        {safe.map(([x, y], i) => (
          <path key={`s${i}`} d={hex(x, y, 7)} className="fill-stage-3/40 stroke-stage-3" />
        ))}
      </g>

      {/* Ranked leads */}
      {leads.map((y, i) => (
        <g key={y}>
          <path d={hex(432, y, 15)} className="fill-stage-4 stroke-stage-4" strokeWidth="1.2" />
          <text
            x="432"
            y={y + 4.2}
            textAnchor="middle"
            className="fill-accent-on font-display"
            style={{ fontSize: 13, fontWeight: 600 }}
          >
            {i + 1}
          </text>
        </g>
      ))}

      {/* Column captions */}
      {COLUMNS.map((c) => (
        <text
          key={c.label}
          x={c.x}
          y="268"
          textAnchor="middle"
          className="fill-ink-3 font-sans"
          style={{ fontSize: 11, fontWeight: 500, letterSpacing: '0.08em' }}
        >
          {c.label.toUpperCase()}
        </text>
      ))}
    </svg>
  )
}
