import { useEffect, useRef, useState } from 'react'

export function usePrefersReducedMotion(): boolean {
  const [reduced, setReduced] = useState(
    () => window.matchMedia('(prefers-reduced-motion: reduce)').matches,
  )
  useEffect(() => {
    const mq = window.matchMedia('(prefers-reduced-motion: reduce)')
    const onChange = () => setReduced(mq.matches)
    mq.addEventListener('change', onChange)
    return () => mq.removeEventListener('change', onChange)
  }, [])
  return reduced
}

/**
 * Animates a readout from its previous value to the next one.
 *
 * Counting is honest here in a way a fake progress bar is not: every value it
 * lands on was actually produced by the backend, and the tween only covers the
 * gap between two real readings. It resolves instantly under reduced motion.
 */
export function useCountUp(target: number, durationMs = 900): number {
  const reduced = usePrefersReducedMotion()
  const [value, setValue] = useState(target)
  const fromRef = useRef(target)
  const frameRef = useRef(0)

  useEffect(() => {
    if (reduced || !Number.isFinite(target)) {
      setValue(target)
      fromRef.current = target
      return
    }

    const from = fromRef.current
    if (from === target) return

    const start = performance.now()
    const tick = (now: number) => {
      const t = Math.min(1, (now - start) / durationMs)
      // Quintic ease-out: fast commitment, long settle — reads as a value
      // arriving rather than a bar filling.
      const eased = 1 - Math.pow(1 - t, 5)
      setValue(from + (target - from) * eased)
      if (t < 1) {
        frameRef.current = requestAnimationFrame(tick)
      } else {
        fromRef.current = target
      }
    }

    frameRef.current = requestAnimationFrame(tick)
    return () => cancelAnimationFrame(frameRef.current)
  }, [target, durationMs, reduced])

  return value
}
