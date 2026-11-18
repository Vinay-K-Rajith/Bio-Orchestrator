import type { ReactNode, SVGProps } from 'react'

/**
 * One icon family for the whole app: a 20px grid, 1.5px round strokes, no
 * fills except for point marks. Icons inherit `currentColor`, so their colour
 * is always decided by the text token around them, never baked in here.
 */

type IconProps = SVGProps<SVGSVGElement>

function Icon({ children, ...props }: IconProps & { children: ReactNode }) {
  return (
    <svg
      viewBox="0 0 20 20"
      fill="none"
      stroke="currentColor"
      strokeWidth="1.5"
      strokeLinecap="round"
      strokeLinejoin="round"
      aria-hidden="true"
      className="h-4 w-4"
      {...props}
    >
      {children}
    </svg>
  )
}

export const IconBrief = (p: IconProps) => (
  <Icon {...p}>
    <path d="M3.5 5.5h13M3.5 9.5h9M3.5 13.5h4.5" />
    <path d="m11.2 16.8.5-2.3 4.6-4.6a1.2 1.2 0 0 1 1.7 1.7l-4.6 4.6-2.2.6Z" />
  </Icon>
)

export const IconFunnel = (p: IconProps) => (
  <Icon {...p}>
    <path d="M3 4h14l-5.3 6.3v4.9l-3.4 1.8v-6.7L3 4Z" />
  </Icon>
)

export const IconReport = (p: IconProps) => (
  <Icon {...p}>
    <path d="M5.3 2.5h6.4l3.8 3.8v10.4c0 .5-.4.8-.8.8H5.3a.8.8 0 0 1-.8-.8V3.3c0-.5.4-.8.8-.8Z" />
    <path d="M11.5 2.6v3.9h3.9M7.5 10.5h5M7.5 13.5h3.5" />
  </Icon>
)

export const IconEvidence = (p: IconProps) => (
  <Icon {...p}>
    <circle cx="8.6" cy="8.6" r="5.4" />
    <path d="m12.6 12.6 4.4 4.4M6.4 7.4h4.4M6.4 9.9h2.8" />
  </Icon>
)

/** A benzene ring: the lead compounds themselves. */
export const IconMolecule = (p: IconProps) => (
  <Icon {...p}>
    <path d="M10 3.2 15.9 6.6v6.8L10 16.8 4.1 13.4V6.6L10 3.2Z" />
    <circle cx="10" cy="10" r="2.9" />
  </Icon>
)

export const IconParse = (p: IconProps) => (
  <Icon {...p}>
    <path d="M7 5 3 10l4 5M13 5l4 5-4 5M11.2 4.5l-2.4 11" />
  </Icon>
)

export const IconScore = (p: IconProps) => (
  <Icon {...p}>
    <circle cx="10" cy="10" r="7" />
    <circle cx="10" cy="10" r="3.6" />
    <circle cx="10" cy="10" r="0.9" fill="currentColor" stroke="none" />
  </Icon>
)

export const IconShield = (p: IconProps) => (
  <Icon {...p}>
    <path d="M10 2.8 16 5v4.6c0 3.7-2.5 6.3-6 7.6-3.5-1.3-6-3.9-6-7.6V5l6-2.2Z" />
    <path d="m7.4 10.1 1.8 1.8 3.5-3.6" />
  </Icon>
)

export const IconRank = (p: IconProps) => (
  <Icon {...p}>
    <path d="M3 16.5h14M4.5 16.5V12h3.5v4.5M8 16.5V7.5h4v9M12 16.5v-6h3.5v6" />
  </Icon>
)

/**
 * The product mark: a ring (the molecule) with a hub wired to three of its
 * atoms (the orchestration). Drawn on its own tile so it holds up at favicon
 * size, where the same geometry is reused in /public/favicon.svg.
 */
export function LogoMark({ className = 'h-8 w-8' }: { className?: string }) {
  return (
    <svg viewBox="0 0 32 32" className={className} aria-hidden="true">
      <rect width="32" height="32" rx="8" className="fill-accent" />
      <g
        fill="none"
        strokeWidth="1.7"
        strokeLinecap="round"
        strokeLinejoin="round"
        className="stroke-accent-on"
      >
        <path d="M16 6.5 24.2 11.25v9.5L16 25.5l-8.2-4.75v-9.5L16 6.5Z" opacity="0.55" />
        <path d="M16 16V6.5M16 16l8.2 4.75M16 16l-8.2 4.75" />
      </g>
      <g className="fill-accent-on">
        <circle cx="16" cy="16" r="2.6" />
        <circle cx="16" cy="6.5" r="1.9" />
        <circle cx="24.2" cy="20.75" r="1.9" />
        <circle cx="7.8" cy="20.75" r="1.9" />
      </g>
    </svg>
  )
}
