import type { Config } from 'tailwindcss'

/**
 * Every colour resolves through a CSS custom property holding `R G B`
 * channels, so a single `data-theme` swap on <html> re-skins the whole app
 * without Tailwind emitting a second copy of every utility. Keeping the
 * channels un-wrapped is what lets `/50` opacity modifiers still work.
 */
const token = (name: string) => `rgb(var(--${name}) / <alpha-value>)`

export default {
  content: ['./index.html', './src/**/*.{ts,tsx}'],
  theme: {
    extend: {
      colors: {
        page: token('page'),
        panel: token('panel'),
        sunken: token('sunken'),
        rule: token('rule'),
        'rule-strong': token('rule-strong'),
        // Input edges only: the one neutral that must clear 3:1 against panel.
        control: token('control'),

        ink: { DEFAULT: token('ink'), 2: token('ink-2'), 3: token('ink-3') },

        accent: {
          DEFAULT: token('accent'),
          hover: token('accent-hover'),
          soft: token('accent-soft'),
          on: token('accent-on'),
        },

        good: { DEFAULT: token('good'), soft: token('good-soft') },
        warn: { DEFAULT: token('warn'), soft: token('warn-soft') },
        crit: { DEFAULT: token('crit'), soft: token('crit-soft') },
        info: { DEFAULT: token('info'), soft: token('info-soft') },

        // Ordinal funnel ramp: one hue, monotone lightness, validated in both
        // themes. Stage order is fixed; the anchor flips for dark.
        stage: {
          1: token('stage-1'),
          2: token('stage-2'),
          3: token('stage-3'),
          4: token('stage-4'),
        },
      },

      fontFamily: {
        display: ['"Fraunces Variable"', 'Iowan Old Style', 'Georgia', 'serif'],
        sans: ['"Instrument Sans Variable"', 'ui-sans-serif', 'system-ui', 'sans-serif'],
        mono: ['"IBM Plex Mono"', 'ui-monospace', 'SFMono-Regular', 'monospace'],
      },

      fontSize: {
        // 12px body-small and 11px caption: the floor for dense UI copy. The
        // previous 11 / 10 px pair put most of the explanatory text below
        // comfortable reading size on a 1440px screen.
        '2xs': ['0.75rem', { lineHeight: '1.125rem', letterSpacing: '0.005em' }],
        micro: ['0.6875rem', { lineHeight: '1rem', letterSpacing: '0.04em' }],
      },

      borderRadius: { panel: '0.625rem', card: '0.875rem' },

      boxShadow: {
        // Tinted toward the accent rather than pure black, so cards sit on the
        // canvas instead of floating over a void.
        card: '0 1px 2px rgb(var(--shadow) / 0.05), 0 8px 24px -12px rgb(var(--shadow) / 0.10)',
        lift: '0 2px 4px rgb(var(--shadow) / 0.06), 0 16px 40px -16px rgb(var(--shadow) / 0.16)',
        inset: 'inset 0 1px 0 rgb(var(--panel) / 0.6)',
      },

      transitionTimingFunction: {
        // A slight overshoot-free ease that settles rather than snaps.
        settle: 'cubic-bezier(0.16, 1, 0.3, 1)',
        precise: 'cubic-bezier(0.4, 0, 0.2, 1)',
      },

      keyframes: {
        'rise-in': {
          from: { opacity: '0', transform: 'translateY(10px)' },
          to: { opacity: '1', transform: 'translateY(0)' },
        },
        'fade-in': { from: { opacity: '0' }, to: { opacity: '1' } },
        'sweep': {
          // The stage indicator: a band that traverses once per active stage.
          '0%': { transform: 'translateX(-100%)' },
          '100%': { transform: 'translateX(400%)' },
        },
        'pulse-dot': {
          '0%, 100%': { opacity: '1', transform: 'scale(1)' },
          '50%': { opacity: '0.45', transform: 'scale(0.82)' },
        },
        flow: {
          // Dashes travelling along the illustration's connectors.
          to: { strokeDashoffset: '-24' },
        },
        'draw-in': {
          from: { opacity: '0', transform: 'scale(0.97)' },
          to: { opacity: '1', transform: 'scale(1)' },
        },
      },

      animation: {
        'rise-in': 'rise-in 0.5s cubic-bezier(0.16, 1, 0.3, 1) both',
        'fade-in': 'fade-in 0.4s ease-out both',
        sweep: 'sweep 1.6s cubic-bezier(0.4, 0, 0.2, 1) infinite',
        'pulse-dot': 'pulse-dot 1.8s ease-in-out infinite',
        flow: 'flow 1.4s linear infinite',
        'draw-in': 'draw-in 0.45s cubic-bezier(0.16, 1, 0.3, 1) both',
      },
    },
  },
  plugins: [],
} satisfies Config
