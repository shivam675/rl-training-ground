import { useState } from 'react'
import { useStore } from '../store'

export default function Robot() {
  const robot = useStore((s) => s.robotInfo)
  const dyn = robot?.dynamics
  const busy = useStore((s) => s.busy)
  const loadUrdf = useStore((s) => s.loadUrdf)
  const reset = useStore((s) => s.resetSimulation)
  const fix = useStore((s) => s.fixRobotDynamics)
  const setGravity = useStore((s) => s.setGravity)

  const [path, setPath] = useState('')
  const [pos, setPos] = useState([0, 0, 0.5])
  const [fixedBase, setFixedBase] = useState(false)
  const [addPlane, setAddPlane] = useState(true)
  const [grav, setGrav] = useState(-9.81)

  const joints: any[] = robot?.joints ?? []
  const hasIssues = dyn && ((dyn.error_count ?? 0) > 0 || (dyn.warning_count ?? 0) > 0)

  return (
    <div className="scroll">
      <div className="page-title">Robot Setup</div>
      <div className="page-sub">Load a URDF and inspect its joints.</div>

      <div className="card">
        <div className="card-head"><span className="ico">📂</span>Load URDF</div>
        <div className="card-body col">
          <label className="field">Server-side URDF path
            <input value={path} placeholder="C:\path\to\robot.urdf" onChange={(e) => setPath(e.target.value)} />
          </label>
          <div className="grid3">
            {['x', 'y', 'z'].map((ax, i) => (
              <label key={ax} className="field">base {ax}
                <input type="number" step={0.1} value={pos[i]} onChange={(e) => setPos(pos.map((v, j) => (j === i ? +e.target.value : v)))} />
              </label>
            ))}
          </div>
          <div className="row wrap">
            <label className="check"><input type="checkbox" checked={fixedBase} onChange={(e) => setFixedBase(e.target.checked)} />Fixed base</label>
            <label className="check"><input type="checkbox" checked={addPlane} onChange={(e) => setAddPlane(e.target.checked)} />Add ground plane</label>
          </div>
          <div className="row">
            <button className="btn primary" disabled={busy || !path.trim()} onClick={() => loadUrdf({ path, basePosition: pos, fixedBase, addPlane })}>Load</button>
            <button className="btn" disabled={busy} onClick={reset}>Reset sim</button>
          </div>
        </div>
      </div>

      {hasIssues && (
        <div className="card">
          <div className="card-body">
            <div className="banner warn">
              <span className="grow">{dyn.summary ?? `${dyn.error_count ?? 0} errors, ${dyn.warning_count ?? 0} warnings in robot dynamics.`}</span>
            </div>
            {(dyn.fixable_count ?? 0) > 0 && <button className="btn" disabled={busy} onClick={fix}>Auto-fix dynamics</button>}
          </div>
        </div>
      )}

      <div className="card">
        <div className="card-head"><span className="ico">🌍</span>Simulation</div>
        <div className="card-body row">
          <label className="field grow">Gravity Z
            <input type="number" step={0.1} value={grav} onChange={(e) => setGrav(+e.target.value)} />
          </label>
          <button className="btn" style={{ alignSelf: 'flex-end' }} disabled={busy} onClick={() => setGravity(grav)}>Apply</button>
        </div>
      </div>

      {robot?.name && (
        <div className="card">
          <div className="card-head"><span className="ico">🤖</span>{robot.name} · {joints.length} joints</div>
          <div className="card-body" style={{ maxHeight: 360, overflowY: 'auto', padding: 0 }}>
            <table className="tbl">
              <thead><tr><th>#</th><th>Name</th><th>Type</th><th>Lower</th><th>Upper</th></tr></thead>
              <tbody>
                {joints.map((j) => (
                  <tr key={j.index ?? j.name}>
                    <td>{j.index}</td><td className="mono">{j.name}</td><td className="muted">{j.type ?? ''}</td>
                    <td>{num(j.lower_limit)}</td><td>{num(j.upper_limit)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}
    </div>
  )
}

const num = (v: any) => (typeof v === 'number' ? v.toFixed(2) : '—')
