import { useStore } from '../store'

export default function ObsAction() {
  const config = useStore((s) => s.envConfig)
  const catalog = useStore((s) => s.observations)
  const obsSize = useStore((s) => s.obsVectorSize)
  const actSize = useStore((s) => s.actionVectorSize)
  const busy = useStore((s) => s.busy)
  const patch = useStore((s) => s.patchConfig)
  const zero = useStore((s) => s.zeroAction)
  const random = useStore((s) => s.randomAction)
  const problems = useStore((s) => s.configProblems)

  const obs: any[] = config?.observations ?? []
  const acts: any[] = config?.actions ?? []
  const meta: Record<string, any> = {}
  for (const s of catalog?.sources ?? []) meta[s.key] = s

  const toggleObs = (key: string, enabled: boolean) =>
    patch({ observations: obs.map((o) => (o.key === key ? { ...o, enabled } : o)) })
  const toggleAct = (idx: number, enabled: boolean) =>
    patch({ actions: acts.map((a) => (a.joint_index === idx ? { ...a, enabled } : a)) })

  return (
    <div className="scroll">
      <div className="page-title">Observation / Action Space</div>
      <div className="page-sub">Pick what the policy senses and controls.</div>
      {problems.map((p) => <div key={p} className="banner err">{p}</div>)}

      <div className="card">
        <div className="card-head"><span className="ico">👁️</span>Observations<span className="grow" /><span className="tag">{obsSize} dims</span></div>
        <div className="card-body col" style={{ gap: 0 }}>
          {obs.length === 0 && <div className="muted">Load a robot to populate observations.</div>}
          {obs.map((o) => {
            const m = meta[o.key] ?? {}
            return (
              <label key={o.key} className="check row spread" style={{ borderBottom: '1px solid var(--outline)' }}>
                <span className="row" style={{ gap: 9 }}>
                  <input type="checkbox" checked={!!o.enabled} disabled={busy} onChange={(e) => toggleObs(o.key, e.target.checked)} />
                  <span>
                    <b>{m.label ?? o.key}</b>
                    {m.description && <div className="faint" style={{ fontSize: 11 }}>{m.description}</div>}
                  </span>
                </span>
                {m.dim != null && <span className="tag">{m.dim}</span>}
              </label>
            )
          })}
        </div>
      </div>

      <div className="card">
        <div className="card-head"><span className="ico">🎛️</span>Actions (joints)<span className="grow" /><span className="tag">{actSize} dims</span></div>
        <div className="card-body col" style={{ gap: 0 }}>
          {acts.length === 0 && <div className="muted">No controllable joints.</div>}
          {acts.map((a) => (
            <label key={a.joint_index} className="check row spread" style={{ borderBottom: '1px solid var(--outline)' }}>
              <span className="row" style={{ gap: 9 }}>
                <input type="checkbox" checked={!!a.enabled} disabled={busy} onChange={(e) => toggleAct(a.joint_index, e.target.checked)} />
                <span><b className="mono">{a.joint_name ?? `joint ${a.joint_index}`}</b>
                  <div className="faint" style={{ fontSize: 11 }}>{a.control_mode ?? 'position'} · [{num(a.lower_limit)}, {num(a.upper_limit)}]</div>
                </span>
              </span>
            </label>
          ))}
          <div className="row" style={{ marginTop: 12 }}>
            <button className="btn sm" disabled={busy || !actSize} onClick={zero}>Zero action</button>
            <button className="btn sm" disabled={busy || !actSize} onClick={random}>Random action</button>
          </div>
        </div>
      </div>
    </div>
  )
}

const num = (v: any) => (typeof v === 'number' ? v.toFixed(2) : '—')
