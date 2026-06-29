import { postJson } from '../api'
import { Viewport } from '../components/Viewport'
import { Chart } from '../components/Chart'
import {
  useStore, hasRobot, hasEnabledObservations, hasEnabledActions, hasEnabledRewards,
  canStartTraining, trainingBlockers, trainingActive, anyJobActive,
} from '../store'
import { useUi } from '../ui'

const GOALS = ['walk', 'balance', 'stand', 'reach a target']

export default function Home() {
  const robot = useStore((s) => s.robotInfo)
  const problems = useStore((s) => s.configProblems)
  const warnings = useStore((s) => s.configWarnings)
  const loadEnvConfig = useStore((s) => s.loadEnvConfig)
  const robotReady = useStore(hasRobot)
  const setPage = useUi((s) => s.setPage)

  const steps = [
    { ok: useStore(hasRobot), label: 'Robot loaded' },
    { ok: useStore(hasEnabledObservations), label: 'Observations enabled' },
    { ok: useStore(hasEnabledActions), label: 'Actions enabled' },
    { ok: useStore(hasEnabledRewards), label: 'Reward set' },
    { ok: useStore((s) => s.configProblems.length === 0), label: 'Config valid' },
  ]
  const done = steps.filter((s) => s.ok).length

  const applyGoal = async (goal: string) => {
    try { await postJson('/reward/apply_goal', { goal }) } catch {}
    await loadEnvConfig()
    setPage('rewards')
  }

  return (
    <div className="split" style={{ padding: 14 }}>
      <div style={{ flex: 1.5, minWidth: 0, display: 'flex' }}>
        <div className="card grow" style={{ display: 'flex', flexDirection: 'column', marginBottom: 0 }}>
          <Viewport />
        </div>
      </div>

      <div style={{ flex: 1, minWidth: 320, overflowY: 'auto' }}>
        <div className="card">
          <div className="card-head"><span className="ico">📋</span>Dashboard</div>
          <div className="card-body">
            {robot?.name ? (
              <div className="muted" style={{ marginBottom: 10 }}>
                <b style={{ color: 'var(--text)' }}>{robot.name}</b> · {robot.joint_count ?? 0} joints
                <div className="faint mono" style={{ fontSize: 11 }}>{robot.path ?? ''}</div>
              </div>
            ) : <div className="muted" style={{ marginBottom: 10 }}>No robot loaded.</div>}

            <div className="row spread" style={{ marginBottom: 6 }}>
              <span className="muted">Setup {done}/5</span>
              <span className={done === 5 ? '' : 'faint'} style={{ color: done === 5 ? 'var(--success)' : undefined }}>{done === 5 ? 'Ready to train' : 'Incomplete'}</span>
            </div>
            <div className="bar" style={{ marginBottom: 12 }}><i style={{ width: `${(done / 5) * 100}%`, background: done === 5 ? 'var(--success)' : 'var(--accent)' }} /></div>

            <div className="col" style={{ gap: 5, marginBottom: 12 }}>
              {steps.map((s) => (
                <div key={s.label} className="row" style={{ gap: 8 }}>
                  <span style={{ color: s.ok ? 'var(--success)' : 'var(--text-faint)' }}>{s.ok ? '✓' : '○'}</span>
                  <span className={s.ok ? '' : 'muted'}>{s.label}</span>
                </div>
              ))}
            </div>

            {problems.map((p) => <div key={p} className="banner err">{p}</div>)}
            {warnings.map((w) => <div key={w} className="banner warn">{w}</div>)}

            <div className="muted" style={{ fontSize: 12, margin: '8px 0 6px' }}>Suggest a goal:</div>
            <div className="row wrap" style={{ gap: 6 }}>
              {GOALS.map((g) => (
                <button key={g} className="chip" style={{ cursor: 'pointer' }} disabled={!robotReady} onClick={() => applyGoal(g)}>{g}</button>
              ))}
            </div>
          </div>
        </div>

        <TrainingCard />
      </div>
    </div>
  )
}

function TrainingCard() {
  const locked = !useStore(canStartTraining)
  const blockers = useStore(trainingBlockers)
  const active = useStore(trainingActive)
  const jobActive = useStore(anyJobActive)
  const status = useStore((s) => s.trainingStatus)
  const telemetry = useStore((s) => s.telemetry)
  const algo = useStore((s) => s.trainingAlgorithm)
  const steps = useStore((s) => s.trainingTimesteps)
  const set = useStore((s) => s.set)
  const start = useStore((s) => s.startTraining)
  const stop = useStore((s) => s.stopTraining)
  const setPage = useUi((s) => s.setPage)

  if (locked)
    return (
      <div className="card">
        <div className="card-body row spread">
          <span className="muted">Finish setup to train: <b>{blockers[0]}</b></span>
          <button className="btn sm" onClick={() => setPage('robot')}>Go</button>
        </div>
      </div>
    )

  const pct = status?.total_timesteps ? Math.min(100, (status.timestep / status.total_timesteps) * 100) : 0

  return (
    <div className="card">
      <div className="card-head"><span className="ico">📈</span>Training<span className="grow" /><button className="btn ghost sm" onClick={() => setPage('training')}>Full controls</button></div>
      <div className="card-body">
        <div className="row" style={{ marginBottom: 10 }}>
          <select value={algo} onChange={(e) => set({ trainingAlgorithm: e.target.value })} disabled={active}>
            {['PPO', 'SAC', 'TD3', 'DQN', 'A2C'].map((a) => <option key={a}>{a}</option>)}
          </select>
          <input className="grow" type="number" value={steps} min={1000} step={1000} onChange={(e) => set({ trainingTimesteps: +e.target.value })} disabled={active} />
          {active
            ? <button className="btn danger" onClick={stop}>Stop</button>
            : <button className="btn primary" disabled={jobActive} onClick={start}>Start</button>}
        </div>

        {active && (
          <>
            <div className="row wrap" style={{ gap: 6, marginBottom: 8 }}>
              <span className="chip"><span className="k">step</span> {status?.timestep ?? 0}/{status?.total_timesteps ?? 0}</span>
              <span className="chip"><span className="k">reward</span> {fmt(status?.episode_reward)}</span>
              <span className="chip"><span className="k">fps</span> {fmt(status?.fps)}</span>
            </div>
            <div className="bar" style={{ marginBottom: 10 }}><i style={{ width: `${pct}%` }} /></div>
          </>
        )}

        {telemetry.length > 0 && <Chart points={telemetry} />}
      </div>
    </div>
  )
}

export const fmt = (v: any) => (typeof v === 'number' ? (Math.abs(v) >= 100 ? v.toFixed(0) : v.toFixed(2)) : '—')
