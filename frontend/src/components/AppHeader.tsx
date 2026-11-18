import { useTheme } from '@/lib/theme'
import type { Health } from '@/types/api'
import { fixed, int } from '@/lib/format'
import { LogoMark } from './icons'

/**
 * The masthead carries the model's own accuracy figures, unprompted.
 *
 * A screening tool that shows predictions but keeps its scaffold-split R² on
 * another page invites the reader to trust a number they have no calibration
 * for. Putting it in the header costs one line and sets the right expectation
 * before the first result appears.
 */

function ThemeToggle() {
  const { theme, toggle } = useTheme()
  const dark = theme === 'dark'

  return (
    <button
      type="button"
      onClick={toggle}
      className="btn-quiet !px-2.5 !py-2"
      aria-label={`Switch to ${dark ? 'light' : 'dark'} theme`}
      title={`Switch to ${dark ? 'light' : 'dark'} theme`}
    >
      <svg
        viewBox="0 0 20 20"
        className="h-4 w-4 transition-transform duration-500 ease-settle"
        style={{ transform: `rotate(${dark ? 180 : 0}deg)` }}
        fill="none"
        stroke="currentColor"
        strokeWidth="1.4"
        strokeLinecap="round"
      >
        {dark ? (
          <path d="M15.5 12.4A6.2 6.2 0 0 1 7.6 4.5a6.5 6.5 0 1 0 7.9 7.9Z" />
        ) : (
          <>
            <circle cx="10" cy="10" r="3.6" />
            <path d="M10 2.2v1.6M10 16.2v1.6M17.8 10h-1.6M3.8 10H2.2M15.5 4.5l-1.1 1.1M5.6 14.4l-1.1 1.1M15.5 15.5l-1.1-1.1M5.6 5.6 4.5 4.5" />
          </>
        )}
      </svg>
    </button>
  )
}

function HeaderStat({
  label,
  value,
  tone = 'text-ink',
  hint,
}: {
  label: string
  value: string
  tone?: string
  hint: string
}) {
  return (
    <div title={hint} className="text-right">
      <div className="label">{label}</div>
      <div className={`numeral text-[0.8125rem] font-medium ${tone}`}>{value}</div>
    </div>
  )
}

export default function AppHeader({ health }: { health: Health | null }) {
  const models = health?.models ?? null
  const s = models?.surrogate

  return (
    <header className="sticky top-0 z-30 border-b border-rule bg-panel/80 backdrop-blur-md">
      <div className="mx-auto flex max-w-[84rem] items-center gap-6 px-6 py-3 sm:px-8">
        <LogoMark className="h-9 w-9 shrink-0" />
        <div className="min-w-0 flex-1">
          <h1 className="font-display text-[1.35rem] font-semibold leading-none tracking-[-0.015em] text-ink">
            Bio<span className="text-accent">Orchestrator</span>
          </h1>
          <p className="mt-1.5 text-2xs text-ink-3">
            Edge-native virtual screening
            <span className="mx-1.5 text-rule-strong">·</span>
            human EGFR
            <span className="mx-1.5 text-rule-strong">·</span>
            ChEMBL 37
          </p>
        </div>

        {s && models && (
          <div className="hidden items-center gap-6 border-r border-rule pr-6 md:flex">
            <HeaderStat
              label="Pool"
              value={int(models.pool_size)}
              hint="Compounds scored at startup and held in memory"
            />
            <HeaderStat
              label="Scaffold R²"
              value={fixed(s.scaffold_r2, 3)}
              tone="text-accent"
              hint="Accuracy on unseen chemical scaffolds — the honest figure"
            />
            <HeaderStat
              label="Coverage"
              value={fixed(s.conformal_coverage, 3)}
              tone="text-good"
              hint="Empirical coverage of the 90% prediction intervals"
            />
            <HeaderStat
              label="hERG"
              value={models.herg_model ? 'loaded' : 'absent'}
              tone={models.herg_model ? 'text-good' : 'text-ink-3'}
              hint="Cardiotoxicity classifier availability"
            />
          </div>
        )}

        <ThemeToggle />
      </div>
    </header>
  )
}
