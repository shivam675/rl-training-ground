import { useState } from 'react'
import { postJson } from '../api'
import { useStore } from '../store'
import { useUi } from '../ui'

const GOALS = ['walk', 'run', 'stand', 'balance', 'jump']

export default function Rewards() {
  const config = useStore((s) => s.envConfig)
  const busy = useStore((s) => s.busy)
  const patch = useStore((s) => s.patchConfig)
  const test = useStore((s) => s.testReward)
  const result = useStore((s) => s.lastRewardResult)
  const validate = useStore((s) => s.validateCustomReward)
  const loadEnvConfig = useStore((s) => s.loadEnvConfig)
  const problems = useStore((s) => s.configProblems)
  const setPage = useUi((s) => s.setPage)

  const rewards: any[] = config?.rewards ?? []
  const [validation, setValidation] = useState<Record<string, any>>({})

  const update = (key: string, fields: any) =>
    patch({ rewards: rewards.map((r) => (r.key === key ? { ...r, ...fields } : r)) })
  const applyGoal = async (goal: string) => { try { await postJson('/reward/apply_goal', { goal }) } catch {} await loadEnvConfig() }
  const checkCode = async (key: string, code: string) => { const r = await validate(code); setValidation((v) => ({ ...v, [key]: r })) }

  return (
    <div className="scroll">
      <div className="page-title">Reward Builder</div>
      <div className="page-sub">Shape what the policy is rewarded for.</div>
      {problems.map((p) => <div key={p} className="banner err">{p}</div>)}

      <div className="card">
        <div className="card-body row wrap" style={{ gap: 7 }}>
          <span className="muted" style={{ alignSelf: 'center' }}>Quick goal:</span>
          {GOALS.map((g) => <button key={g} className="chip" style={{ cursor: 'pointer' }} disabled={busy} onClick={() => applyGoal(g)}>{g}</button>)}
          <span className="grow" />
          <button className="btn primary" disabled={busy} onClick={test}>Test reward</button>
        </div>
      </div>

      {result && (
        <div className="card">
          <div className="card-head"><span className="ico">🧮</span>Result</div>
          <div className="card-body">
            <div className="row spread" style={{ marginBottom: 8 }}>
              <span className="muted">total reward</span>
              <b style={{ fontSize: 18, color: 'var(--accent)' }}>{fmt(result.reward)}</b>
            </div>
            {result.formula && <div className="mono faint" style={{ fontSize: 12, marginBottom: 8 }}>{result.formula}</div>}
            {Array.isArray(result.terms) && result.terms.length > 0 && (
              <table className="tbl">
                <thead><tr><th>term</th><th>value</th></tr></thead>
                <tbody>{result.terms.map((t: any, i: number) => (
                  <tr key={i}><td className="mono">{t.key ?? t.name ?? JSON.stringify(t)}</td><td>{fmt(t.value ?? t.contribution)}</td></tr>
                ))}</tbody>
              </table>
            )}
          </div>
        </div>
      )}

      <div className="card">
        <div className="card-head"><span className="ico">🏆</span>Components</div>
        <div className="card-body col" style={{ gap: 0 }}>
          {rewards.length === 0 && <div className="muted">Pick a goal or load a robot to populate reward components.</div>}
          {rewards.map((r) => (
            <div key={r.key} style={{ borderBottom: '1px solid var(--outline)', padding: '10px 0' }}>
              <div className="row spread">
                <label className="check" style={{ padding: 0 }}>
                  <input type="checkbox" checked={!!r.enabled} disabled={busy} onChange={(e) => update(r.key, { enabled: e.target.checked })} />
                  <b>{r.key}</b>
                </label>
                <label className="row" style={{ gap: 6 }}>
                  <span className="faint" style={{ fontSize: 11 }}>weight</span>
                  <input type="number" step={0.1} defaultValue={r.weight ?? 1} style={{ width: 80 }} disabled={busy}
                    onBlur={(e) => +e.target.value !== r.weight && update(r.key, { weight: +e.target.value })} />
                </label>
              </div>
              {(r.key === 'custom' || r.params?.code != null) && (
                <div className="col" style={{ marginTop: 8 }}>
                  <textarea rows={4} className="mono" defaultValue={r.params?.code ?? ''} placeholder="# return float reward"
                    onBlur={(e) => update(r.key, { params: { ...r.params, code: e.target.value } })} />
                  <div className="row">
                    <button className="btn sm" disabled={busy} onClick={(e) => checkCode(r.key, (e.currentTarget.closest('.col')!.querySelector('textarea') as HTMLTextAreaElement).value)}>Validate</button>
                    {validation[r.key] && <span className={validation[r.key].ok === false || validation[r.key].errors?.length ? 'banner err' : 'banner ok'} style={{ margin: 0, padding: '4px 8px' }}>
                      {(validation[r.key].errors?.length ? validation[r.key].errors : validation[r.key].warnings ?? ['valid']).join('; ')}
                    </span>}
                  </div>
                </div>
              )}
            </div>
          ))}
          <button className="btn ghost sm" style={{ marginTop: 10, alignSelf: 'flex-start' }} onClick={() => setPage('training')}>Continue to Training →</button>
        </div>
      </div>
    </div>
  )
}

const fmt = (v: any) => (typeof v === 'number' ? v.toFixed(3) : '—')
