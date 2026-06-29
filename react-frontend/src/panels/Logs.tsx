import { useEffect, useRef, useState } from 'react'
import { useStore } from '../store'

export default function Logs() {
  const fetchLogs = useStore((s) => s.fetchBackendLogs)
  const [lines, setLines] = useState<string[]>([])
  const [follow, setFollow] = useState(true)
  const ref = useRef<HTMLPreElement>(null)

  useEffect(() => {
    let alive = true
    const tick = async () => { const l = await fetchLogs(600); if (alive) setLines(l) }
    tick()
    const id = setInterval(tick, 3000)
    return () => { alive = false; clearInterval(id) }
  }, [fetchLogs])

  useEffect(() => { if (follow && ref.current) ref.current.scrollTop = ref.current.scrollHeight }, [lines, follow])

  return (
    <div className="scroll">
      <div className="row spread">
        <div><div className="page-title">Logs</div><div className="page-sub">Live backend output (refreshes every 3s).</div></div>
        <label className="check"><input type="checkbox" checked={follow} onChange={(e) => setFollow(e.target.checked)} />Follow</label>
      </div>
      <div className="card">
        <pre ref={ref} className="mono" style={{ margin: 0, padding: 14, fontSize: 11.5, lineHeight: 1.5, maxHeight: '70vh', overflow: 'auto', whiteSpace: 'pre-wrap' }}>
          {lines.length ? lines.join('\n') : 'No log output.'}
        </pre>
      </div>
    </div>
  )
}
