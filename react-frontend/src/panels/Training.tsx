import { useState } from 'react'
import { Chart } from '../components/Chart'
import { fmt } from './Home'
import {
  useStore, trainingBlockers, canStartTraining, trainingActive, tuningActive,
  anyJobActive, blockingJobLabel,
} from '../store'

const ALGOS = ['PPO', 'SAC', 'TD3', 'DQN', 'A2C']
const HP: { key: string; label: string; step?: number }[] = [
  { key: 'learning_rate', label: 'learning rate', step: 0.0001 },
  { key: 'batch_size', label: 'batch size', step: 1 },
  { key: 'gamma', label: 'gamma', step: 0.01 },
  { key: 'n_steps', label: 'n_steps', step: 1 },
  { key: 'ent_coef', label: 'ent_coef', step: 0.001 },
  { key: 'clip_range', label: 'clip_range', step: 0.01 },
  { key: 'tau', label: 'tau', step: 0.001 },
  { key: 'buffer_size', label: 'buffer_size', step: 1000 },
  { key: 'train_freq', label: 'train_freq', step: 1 },
  { key: 'seed', label: 'seed', step: 1 },
]

export default function Training() {
  const blockers = useStore(trainingBlockers)
  const locked = !useStore(canStartTraining)
  const active = useStore(trainingActive)
  const jobActive = useStore(anyJobActive)
  const blockLabel = useStore(blockingJobLabel)
  const status = useStore((s) => s.trainingStatus)
  const telemetry = useStore((s) => s.telemetry)
  const obsSize = useStore((s) => s.obsVectorSize)
  const actSize = useStore((s) => s.actionVectorSize)
  const algo = useStore((s) => s.trainingAlgorithm)
  const steps = useStore((s) => s.trainingTimesteps)
  const params = useStore((s) => s.trainingParams)
  const advisor = useStore((s) => s.advisor)
  const set = useStore((s) => s.set)
  const start = useStore((s) => s.startTraining)
  const stop = useStore((s) => s.stopTraining)

  const [showHp, setShowHp] = useState(false)
  const setParam = (k: string, v: any) => set({ trainingParams: { ...params, [k]: v === '' || v == null ? undefined : v } })
  const netArch = Array.isArray(params.net_arch) ? params.net_arch.join(',') : ''
  const pct = status?.total_timesteps ? Math.min(100, (status.timestep / status.total_timesteps) * 100) : 0

  return (
    <div className="scroll">
      <div className="page-title">Training</div>
      <div className="page-sub">Configure and launch the RL run.</div>

      {locked && <div className="banner err">Locked — {blockers.join(', ')}.</div>}
      {!locked && blockLabel && <div className="banner warn">{blockLabel} is running; only one job at a time.</div>}

      {advisor?.algorithm && (
        <div className="banner info">
          <span className="grow">Recommended: <b>{advisor.algorithm}</b>{advisor.reason ? ` — ${advisor.reason}` : ''}</span>
          <button className="btn sm" onClick={() => { set({ trainingAlgorithm: advisor.algorithm }); if (advisor.hyperparams) set({ trainingParams: { ...params, ...advisor.hyperparams } }) }}>Apply</button>
        </div>
      )}

      <div className="card">
        <div className="card-head"><span className="ico">⚙️</span>Run</div>
        <div className="card-body col">
          <div className="row wrap">
            <label className="field">Algorithm
              <select value={algo} onChange={(e) => set({ trainingAlgorithm: e.target.value })} disabled={active}>{ALGOS.map((a) => <option key={a}>{a}</option>)}</select>
            </label>
            <label className="field grow">Total timesteps
              <input type="number" min={1000} step={1000} value={steps} onChange={(e) => set({ trainingTimesteps: +e.target.value })} disabled={active} />
            </label>
            <label className="field grow">Hidden layers (comma)
              <input key={netArch} defaultValue={netArch} placeholder="64,64" disabled={active}
                onBlur={(e) => { const arr = e.target.value.split(',').map((x) => +x.trim()).filter((n) => n >= 16 && n <= 1024); setParam('net_arch', arr.length ? arr.slice(0, 4) : undefined) }} />
            </label>
          </div>

          <button className="btn ghost sm" style={{ alignSelf: 'flex-start' }} onClick={() => setShowHp(!showHp)}>{showHp ? '▾' : '▸'} Hyperparameters</button>
          {showHp && (
            <div className="grid3">
              {HP.map((h) => (
                <label key={h.key} className="field">{h.label}
                  <input type="number" step={h.step} defaultValue={params[h.key] ?? ''} disabled={active}
                    onBlur={(e) => setParam(h.key, e.target.value === '' ? undefined : +e.target.value)} />
                </label>
              ))}
            </div>
          )}

          <div className="row">
            {active ? <button className="btn danger" onClick={stop}>Stop training</button>
              : <button className="btn primary" disabled={locked || jobActive} onClick={start}>Start training</button>}
            <button className="btn ghost sm" disabled={active} onClick={() => set({ trainingParams: { learning_rate: 0.0003, batch_size: 64, gamma: 0.99, n_steps: 256 } })}>Reset params</button>
          </div>
        </div>
      </div>

      <div className="card">
        <div className="card-head"><span className="ico">📊</span>Live</div>
        <div className="card-body">
          <div className="row wrap" style={{ gap: 6, marginBottom: 10 }}>
            <span className="chip"><span className="k">state</span> {status?.active ? 'training' : status?.message ?? 'idle'}</span>
            <span className="chip"><span className="k">step</span> {status?.timestep ?? 0}/{status?.total_timesteps ?? 0}</span>
            <span className="chip"><span className="k">reward</span> {fmt(status?.episode_reward)}</span>
            <span className="chip"><span className="k">fps</span> {fmt(status?.fps)}</span>
            <span className="chip"><span className="k">obs/act</span> {obsSize}/{actSize}</span>
          </div>
          {status?.active && <div className="bar" style={{ marginBottom: 12 }}><i style={{ width: `${pct}%` }} /></div>}
          {telemetry.length > 0 ? <Chart points={telemetry} /> : <div className="muted">No telemetry yet — start a run.</div>}
        </div>
      </div>

      <TuningCard />
    </div>
  )
}

function TuningCard() {
  const tuning = useStore((s) => s.tuningStatus)
  const active = useStore(tuningActive)
  const jobActive = useStore(anyJobActive)
  const busy = useStore((s) => s.busy)
  const algo = useStore((s) => s.trainingAlgorithm)
  const params = useStore((s) => s.trainingParams)
  const set = useStore((s) => s.set)
  const startTuning = useStore((s) => s.startTuning)
  const stopTuning = useStore((s) => s.stopTuning)
  const [nTrials, setNTrials] = useState(8)
  const [tpt, setTpt] = useState(2000)
  const best = tuning?.result?.best_params ?? tuning?.best_params

  return (
    <div className="card">
      <div className="card-head"><span className="ico">🔬</span>Hyperparameter tuning</div>
      <div className="card-body col">
        <div className="row wrap">
          <label className="field">Trials<input type="number" min={2} value={nTrials} onChange={(e) => setNTrials(+e.target.value)} disabled={active} /></label>
          <label className="field">Steps/trial<input type="number" min={500} step={500} value={tpt} onChange={(e) => setTpt(+e.target.value)} disabled={active} /></label>
          <div style={{ alignSelf: 'flex-end' }}>
            {active ? <button className="btn danger" onClick={stopTuning}>Stop</button>
              : <button className="btn" disabled={jobActive} onClick={() => startTuning({ algorithm: algo, nTrials, timestepsPerTrial: tpt })}>Start tuning</button>}
          </div>
        </div>
        {active && <div className="muted">Trial {tuning?.current_trial ?? '?'} / {nTrials}…</div>}
        {best && (
          <div className="row spread">
            <span className="mono faint" style={{ fontSize: 11 }}>{JSON.stringify(best)}</span>
            <button className="btn sm" disabled={busy} onClick={() => set({ trainingParams: { ...params, ...best } })}>Apply best params</button>
          </div>
        )}
      </div>
    </div>
  )
}
