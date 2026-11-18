import { useEffect, useRef, useState } from 'react'
import SmilesDrawer from 'smiles-drawer'
import { useTheme } from '@/lib/theme'

/**
 * A depiction of the molecule, drawn as SVG rather than canvas so it stays
 * crisp on any display and scales with its card.
 *
 * The heteroatom colours are the app's own palette rather than the library
 * default. Structure drawings are the one place where colour already carries
 * chemical meaning, so letting them keep the stock web-safe reds and blues
 * would put a second, louder palette on the page.
 */

type Depiction = { readonly kind: 'pending' } | { readonly kind: 'ok' } | { readonly kind: 'failed' }

const THEMES = {
  light: {
    FOREGROUND: '#1d2238',
    BACKGROUND: '#f4f6fa',
    C: '#1d2238',
    O: '#b42830',
    N: '#3341aa',
    F: '#167252',
    CL: '#167252',
    BR: '#9a4a0c',
    I: '#6c40aa',
    P: '#9a4a0c',
    S: '#925a08',
    B: '#925a08',
    SI: '#6c40aa',
    H: '#5d647e',
  },
  dark: {
    FOREGROUND: '#d5d9e8',
    BACKGROUND: '#0e111e',
    C: '#d5d9e8',
    O: '#f68680',
    N: '#98a8ff',
    F: '#54c894',
    CL: '#54c894',
    BR: '#e8a060',
    I: '#c0a2f4',
    P: '#e8a060',
    S: '#e8b252',
    B: '#e8b252',
    SI: '#c0a2f4',
    H: '#8a92ae',
  },
} as const

/**
 * One drawer for the whole app. It holds no per-molecule state — the SMILES,
 * the target element and the theme are all arguments to `draw` — so an instance
 * per card would just be several copies of the same option table.
 */
const drawer = new SmilesDrawer.SmiDrawer(
  {
    width: 300,
    height: 190,
    padding: 14,
    bondThickness: 1.1,
    bondLength: 16,
    fontSizeLarge: 6,
    fontSizeSmall: 4,
    compactDrawing: false,
    terminalCarbons: false,
    explicitHydrogens: false,
    themes: THEMES,
  },
)

interface Props {
  smiles: string
  label: string
}

export default function StructureView({ smiles, label }: Props) {
  const { theme } = useTheme()
  const ref = useRef<SVGSVGElement | null>(null)
  const [status, setStatus] = useState<Depiction>({ kind: 'pending' })

  useEffect(() => {
    const svg = ref.current
    if (!svg) return

    // The drawer appends into the element, so clear the previous depiction
    // before a theme change redraws over it.
    svg.replaceChildren()
    setStatus({ kind: 'pending' })

    drawer.draw(
      smiles,
      svg,
      theme,
      () => setStatus({ kind: 'ok' }),
      // An unparseable SMILES is a data problem worth showing, not a crash.
      () => setStatus({ kind: 'failed' }),
    )
  }, [smiles, theme])

  return (
    <figure
      className="structure relative flex min-h-[168px] items-center justify-center
                 overflow-hidden rounded-panel border border-rule bg-sunken px-2 py-1"
    >
      <svg
        ref={ref}
        role="img"
        aria-label={`Skeletal structure of ${label}`}
        viewBox="0 0 300 190"
        className={`h-[168px] w-full transition-[opacity,transform] duration-500 ease-settle ${
          status.kind === 'ok' ? 'scale-100 opacity-100' : 'scale-[0.97] opacity-0'
        }`}
      />
      {status.kind === 'failed' && (
        <figcaption className="absolute inset-0 flex items-center justify-center px-4 text-center text-2xs text-ink-3">
          Structure could not be rendered from this SMILES.
        </figcaption>
      )}
    </figure>
  )
}
