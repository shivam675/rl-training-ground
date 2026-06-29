import { useState } from 'react'
import { Viewport } from '../components/Viewport'
import { fmt } from './Home'
import { useStore, evaluationActive, anyJobActive } from '../store'

export default function Evaluation() {
  const runs = useStore((s) => s.runs)
  const showAll = useStore((s) => s.runsShowAll)
  const setShowAll = useStore((s) => s.setRunsShowAll)
  const evalActive = useStore(evaluationActive)
  const evalStatus = useStore((s) => s.evaluationStatus)
  const jobActive = useStore(anyJobActive)
  const busy = useStore((s) => s.busy)
  const startEval = useStore((s) => s.startEvaluation)
  const stopEval = useStore((s) => s.stopEvaluation)
  const exportRun = useStore((s) => s.exportRun)
  const deleteRun = useStore((s) => s.deleteRun)
  const fetchDetails = useStore((s) => s.fetchRunDetails)

  const [episodes, setEpisodes] = useState(3)
  const [det, setDet] = useState(true)
  const [sel, setSel] = useState<Set<string>>(new Set())
  const [details, setDetails] = useState<any | null>(null)

  const toggleSel = (name: string) => setSel((s) => { const n = new Set(s); n.has(name) ? n.delete(name) : n.add(name); return n })
  const selected = runs.filter((r) => sel.has(r.name))

  return (
    <div className="scroll">
      <div className="page-title">Evaluation</div>
      <div className="page-sub">Replay and score trained runs.</div>

      {evalActive && (
        <div className="card">
          <div className="card-head"><span className="ico">🎬</span>Playback<span className="grow" /><button className="btn danger sm" onClick={stopEval}>Stop</button></div>
          <div className="card-body">
            <div className="muted" style={{ marginBottom: 8 }}>{evalStatus?.message ?? `Episode ${evalStatus?.episode ?? '?'} / ${evalStatus?.episodes ?? episodes}`}</div>
            <div style={{ height: 320, display: 'flex' }}><div className="card grow" style={{ marginBottom: 0, display: 'flex', overflow: 'hidden' }}><Viewport /></div></div>
          </div>
        </div>
      )}

      <div className="card">
        <div className="card-head">
          <span className="ico">🗂️</span>Runs<span className="grow" />
          <label className="check" style={{ padding: 0 }}><input type="checkbox" checked={showAll} onChange={(e) => setShowAll(e.target.checked)} />All projects</label>
        </div>
        <div className="card-body">
          <div className="row wrap" style={{ marginBottom: 10, gap: 8 }}>
            <label className="field">Episodes<input type="number" min={1} value={episodes} onChange={(e) => setEpisodes(+e.target.value)} style={{ width: 80 }} /></label>
            <label className="check" style={{ alignSelf: 'flex-end' }}><input type="checkbox" checked={det} onChange={(e) => setDet(e.target.checked)} />Deterministic</label>
          </div>

          {runs.length === 0 ? <div className="muted">No runs yet. Train a model first.</div> : (
            <table className="tbl">
              <thead><tr><th></th><th>Name</th><th>Algo</th><th>Steps</th><th>Best reward</th><th>Created</th><th></th></tr></thead>
              <tbody>
                {runs.map((r) => (
                  <tr key={r.name} className={sel.has(r.name) ? 'sel' : ''}>
                    <td><input type="checkbox" checked={sel.has(r.name)} onChange={() => toggleSel(r.name)} /></td>
                    <td className="mono" style={{ cursor: 'pointer' }} onClick={async () => setDetails(await fetchDetails(r.name))}>{r.name}</td>
                    <td>{r.algorithm ?? '—'}</td>
                    <td>{r.total_timesteps ?? '—'}</td>
                    <td>{fmt(r.best_reward ?? r.mean_reward)}</td>
                    <td className="faint">{shortDate(r.created)}</td>
                    <td>
                      <div className="row" style={{ gap: 4 }}>
                        <button className="btn sm" disabled={jobActive || busy} onClick={() => startEval(r.name, episodes, det)}>Eval</button>
                        <button className="btn ghost sm" disabled={busy} onClick={() => exportRun(r.name)}>⤓</button>
                        <button className="btn ghost sm" disabled={busy} onClick={() => confirm(`Delete ${r.name}?`) && deleteRun(r.name)}>🗑</button>
                      </div>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          )}
        </div>
      </div>

      {selected.length >= 2 && (
        <div className="card">
          <div className="card-head"><span className="ico">⚖️</span>Compare ({selected.length})</div>
          <div className="card-body" style={{ overflowX: 'auto' }}>
            <table className="tbl">
              <thead><tr><th>metric</th>{selected.map((r) => <th key={r.name} className="mono">{r.name}</th>)}</tr></thead>
              <tbody>
                {['algorithm', 'total_timesteps', 'best_reward', 'mean_reward'].map((k) => (
                  <tr key={k}><td className="muted">{k}</td>{selected.map((r) => <td key={r.name}>{r[k] ?? '—'}</td>)}</tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}

      {details && (
        <div className="card">
          <div className="card-head"><span className="ico">📄</span>{details.name ?? 'Run details'}<span className="grow" /><button className="btn ghost sm" onClick={() => setDetails(null)}>✕</button></div>
          <div className="card-body"><pre className="mono" style={{ fontSize: 11, whiteSpace: 'pre-wrap', maxHeight: 320, overflow: 'auto' }}>{JSON.stringify(details, null, 2)}</pre></div>
        </div>
      )}
    </div>
  )
}

const shortDate = (v: any) => { try { return new Date(v).toLocaleString() } catch { return String(v ?? '') } }
