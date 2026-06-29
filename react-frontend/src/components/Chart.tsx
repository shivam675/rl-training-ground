import { useEffect, useRef } from 'react'
import uPlot from 'uplot'
import 'uplot/dist/uPlot.min.css'

// Live reward/loss curve. uPlot redraws thousands of points at 60fps; we feed it
// the telemetry array and call setData on change — no React work per point.

const cssVar = (n: string) => getComputedStyle(document.documentElement).getPropertyValue(n).trim()

export function Chart({ points }: { points: Record<string, any>[] }) {
  const ref = useRef<HTMLDivElement>(null)
  const plot = useRef<uPlot | null>(null)

  useEffect(() => {
    const el = ref.current!
    const opts: uPlot.Options = {
      width: el.clientWidth || 600,
      height: 240,
      padding: [10, 14, 0, 0],
      legend: { show: true },
      cursor: { drag: { x: true, y: false } },
      scales: { x: { time: false } },
      axes: [
        { stroke: cssVar('--text-dim'), grid: { stroke: cssVar('--outline'), width: 1 }, ticks: { stroke: cssVar('--outline') } },
        { stroke: cssVar('--text-dim'), grid: { stroke: cssVar('--outline'), width: 1 }, ticks: { stroke: cssVar('--outline') } },
      ],
      series: [
        { label: 'step' },
        { label: 'reward', stroke: cssVar('--success'), width: 2 },
        { label: 'loss', stroke: cssVar('--danger'), width: 1.5, scale: 'loss' },
      ],
    }
    plot.current = new uPlot(opts, [[], [], []], el)
    const ro = new ResizeObserver(() => plot.current?.setSize({ width: el.clientWidth, height: 240 }))
    ro.observe(el)
    return () => { ro.disconnect(); plot.current?.destroy(); plot.current = null }
  }, [])

  useEffect(() => {
    if (!plot.current) return
    const xs = points.map((p) => p.timestep ?? 0)
    const reward = points.map((p) => (p.reward ?? null) as number | null)
    const loss = points.map((p) => (p.loss ?? null) as number | null)
    plot.current.setData([xs, reward, loss])
  }, [points])

  return <div ref={ref} style={{ width: '100%' }} />
}
