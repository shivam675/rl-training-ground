import { create } from 'zustand'
import { getJson, postJson, streamPostJson, type Json } from './api'

// Port of the Flutter AppState (ChangeNotifier). One store, slice-subscribed by
// panels so a telemetry tick never re-renders the whole app.

export interface State {
  health: Json | null
  agentHealth: Json | null
  robotInfo: Json | null
  observations: Json | null
  actions: Json | null
  trainingStatus: Json | null
  tuningStatus: Json | null
  evaluationStatus: Json | null
  advisor: Json | null
  agentSettings: Json | null
  appPreferences: Json | null
  envConfig: Json | null
  configProblems: string[]
  configWarnings: string[]
  configRevision: number
  envConfigSaved: boolean
  obsVectorSize: number
  actionVectorSize: number
  runs: Json[]
  runsShowAll: boolean
  telemetry: Json[]
  lastRewardResult: Json | null
  message: string
  busy: boolean
  streamResolutionScale: number
  showInspectorOnDashboard: boolean
  agentAutonomy: 'act' | 'ask'
  robotLoadRevision: number
  lastLoadedRobotPath: string | null
  currentProjectPath: string | null
  trainingAlgorithm: string
  trainingTimesteps: number
  trainingParams: Json
}

const initial: State = {
  health: null, agentHealth: null, robotInfo: null, observations: null, actions: null,
  trainingStatus: null, tuningStatus: null, evaluationStatus: null, advisor: null,
  agentSettings: null, appPreferences: null, envConfig: null,
  configProblems: [], configWarnings: [], configRevision: 0, envConfigSaved: false,
  obsVectorSize: 0, actionVectorSize: 0, runs: [], runsShowAll: false, telemetry: [],
  lastRewardResult: null, message: 'Backend not checked yet.', busy: false,
  streamResolutionScale: 1.0, showInspectorOnDashboard: true, agentAutonomy: 'act',
  robotLoadRevision: 0, lastLoadedRobotPath: null, currentProjectPath: null,
  trainingAlgorithm: 'PPO', trainingTimesteps: 10000,
  trainingParams: { learning_rate: 0.0003, batch_size: 64, gamma: 0.99, n_steps: 256 },
}

interface Actions {
  guard: (fn: () => Promise<void>) => Promise<void>
  refreshAll: () => Promise<void>
  refreshAgentHealth: () => Promise<void>
  loadEnvConfig: (notify?: boolean) => Promise<void>
  applyConfigResponse: (res: Json) => void
  loadRuns: () => Promise<void>
  setRunsShowAll: (v: boolean) => void
  loadAdvisor: () => Promise<void>
  loadUrdf: (p: { path: string; basePosition: number[]; fixedBase: boolean; addPlane: boolean }) => Promise<void>
  fixRobotDynamics: () => Promise<void>
  resetSimulation: () => Promise<void>
  setGravity: (z: number) => Promise<void>
  zeroAction: () => Promise<void>
  randomAction: () => Promise<void>
  patchConfig: (patch: Json) => Promise<void>
  undoConfigChange: () => Promise<Json>
  saveEnvConfig: () => Promise<void>
  validateCustomReward: (code: string) => Promise<Json>
  testReward: () => Promise<void>
  startTraining: () => Promise<void>
  stopTraining: () => Promise<void>
  startTuning: (p: { algorithm: string; nTrials: number; timestepsPerTrial: number }) => Promise<void>
  stopTuning: () => Promise<void>
  startEvaluation: (runName: string, episodes: number, deterministic: boolean) => Promise<void>
  stopEvaluation: () => Promise<void>
  fetchRunDetails: (name: string) => Promise<Json | null>
  exportRun: (name: string) => Promise<string | null>
  deleteRun: (name: string) => Promise<boolean>
  fetchBackendLogs: (lines?: number) => Promise<string[]>
  savePreferences: () => Promise<void>
  saveAgentSettings: (providers: Json) => Promise<void>
  checkModelCapabilities: () => Promise<Json>
  executeAgentTool: (name: string, args: Json) => Promise<Json>
  set: (p: Partial<State>) => void
  init: () => void
  pollTelemetry: () => Promise<void>
}

export type Store = State & Actions

export const useStore = create<Store>((set, get) => ({
  ...initial,
  set: (p) => set(p),

  guard: async (fn) => {
    set({ busy: true })
    try { await fn() } catch (e: any) { set({ message: String(e?.message || e) }) }
    finally { set({ busy: false }) }
  },

  applyConfigResponse: (res) => {
    const sizes = res.vector_sizes
    set({
      envConfig: res.config ?? null,
      envConfigSaved: res.saved === true,
      configProblems: (res.problems ?? []).map(String),
      configWarnings: (res.warnings ?? []).map(String),
      configRevision: res.revision ?? get().configRevision,
      obsVectorSize: sizes?.observation_vector_size ?? get().obsVectorSize,
      actionVectorSize: sizes?.action_vector_size ?? get().actionVectorSize,
    })
  },

  refreshAgentHealth: async () => {
    try { set({ agentHealth: await getJson('/agent/health') }) }
    catch { set({ agentHealth: { ok: false, reachable: false } }) }
  },

  refreshAll: async () => {
    await get().guard(async () => {
      const [health, robotInfo, observations, actions, trainingStatus, agentSettings, appPreferences] =
        await Promise.all([
          getJson('/health'), getJson('/robot/info'), getJson('/robot/observations'),
          getJson('/robot/actions'), getJson('/training/status'), getJson('/agent/providers'),
          getJson('/app/preferences'),
        ])
      set({
        health, robotInfo, observations, actions, trainingStatus, agentSettings, appPreferences,
        streamResolutionScale: clamp(appPreferences?.stream_resolution_scale ?? get().streamResolutionScale, 0.5, 1.5),
        showInspectorOnDashboard: appPreferences?.show_inspector_on_dashboard ?? get().showInspectorOnDashboard,
        agentAutonomy: appPreferences?.agent_autonomy === 'ask' ? 'ask' : 'act',
        message: 'Connected to backend.',
      })
      await get().loadRuns()
      await get().loadEnvConfig(false)
      await get().loadAdvisor()
    })
  },

  loadEnvConfig: async () => {
    try { get().applyConfigResponse(await getJson('/env/config')) } catch {}
  },

  loadRuns: async () => {
    try {
      const pid = currentProjectId(get())
      const scoped = !get().runsShowAll && pid
      const q = scoped ? `?project_id=${encodeURIComponent(pid!)}` : ''
      const res = await getJson(`/runs${q}`)
      set({ runs: res.runs ?? [] })
    } catch {}
  },

  setRunsShowAll: (v) => { set({ runsShowAll: v }); get().loadRuns() },

  loadAdvisor: async () => {
    try { set({ advisor: await getJson('/training/advisor') }) } catch {}
  },

  loadUrdf: async ({ path, basePosition, fixedBase, addPlane }) => {
    await get().guard(async () => {
      await postJson('/simulation/load_urdf', { path, base_position: basePosition, fixed_base: fixedBase, add_plane: addPlane })
      const [robotInfo, observations, actions] = await Promise.all([
        getJson('/robot/info'), getJson('/robot/observations'), getJson('/robot/actions'),
      ])
      set({ robotInfo, observations, actions })
      await get().loadEnvConfig(false)
      set({
        robotLoadRevision: get().robotLoadRevision + 1,
        lastLoadedRobotPath: robotInfo?.source_path ?? robotInfo?.path ?? path,
        message: 'Loaded URDF.',
      })
    })
  },

  fixRobotDynamics: async () => {
    await get().guard(async () => {
      const res = await postJson('/robot/fix_dynamics', {})
      set({ robotInfo: await getJson('/robot/info'), message: res.summary ?? 'Repaired robot dynamics.' })
    })
  },

  resetSimulation: async () => {
    await get().guard(async () => {
      await postJson('/simulation/reset', { reload_current_urdf: true })
      set({ message: 'Simulation reset.' })
    })
  },

  setGravity: async (z) => {
    await get().guard(async () => {
      await postJson('/simulation/set_gravity', { gravity: [0, 0, z] })
      set({ message: 'Gravity updated.' })
    })
  },

  zeroAction: async () => {
    await get().guard(async () => {
      await postJson('/env/action_test', { values: Array(get().actionVectorSize).fill(0) })
      set({ message: 'Applied zero normalized action.' })
    })
  },

  randomAction: async () => {
    const values = Array.from({ length: get().actionVectorSize }, (_, i) => (i % 2 === 0 ? 0.15 : -0.15))
    await get().guard(async () => {
      await postJson('/env/action_test', { values })
      set({ message: 'Applied safe normalized random action.' })
    })
  },

  patchConfig: async (patch) => {
    await get().guard(async () => {
      const res = await postJson('/env/config/patch', { patch, source: 'ui' })
      get().applyConfigResponse(res)
      const probs = get().configProblems
      set({ message: probs.length ? `Config updated with problems: ${probs.join('; ')}` : 'Environment config updated.' })
    })
  },

  undoConfigChange: async () => {
    set({ busy: true })
    try {
      const res = await postJson('/env/config/undo', {})
      get().applyConfigResponse(res)
      const summary = (res.change_set?.summary ?? []).map(String).join(' ')
      set({ message: summary || 'Configuration change undone.' })
      return res
    } catch (e: any) {
      set({ message: String(e?.message || e) })
      return { error: String(e?.message || e) }
    } finally { set({ busy: false }) }
  },

  saveEnvConfig: async () => {
    await get().guard(async () => {
      await postJson('/env/save_config', {})
      await get().loadEnvConfig(false)
      set({ message: 'Configuration valid.' })
    })
  },

  validateCustomReward: (code) => postJson('/reward/validate_custom', { code }),

  testReward: async () => {
    const components = get().envConfig?.rewards ?? []
    await get().guard(async () => {
      const result = await postJson('/reward/test', { components })
      set({ lastRewardResult: result, message: `Reward ${result.reward} | ${result.formula}` })
    })
  },

  startTraining: async () => {
    await get().guard(async () => {
      const blockers = trainingBlockers(get())
      if (blockers.length) { set({ message: `Training locked: ${blockers[0]}` }); return }
      const p: Json = { learning_rate: 0.0003, batch_size: 64, gamma: 0.99, n_steps: 256, ...get().trainingParams }
      const total = get().trainingTimesteps
      const body: Json = {
        algorithm: get().trainingAlgorithm, total_timesteps: total,
        learning_rate: p.learning_rate, batch_size: p.batch_size, gamma: p.gamma, n_steps: p.n_steps,
        policy_type: 'MlpPolicy', checkpoint_every: total >= 5000 ? Math.floor(total / 5) : 0,
      }
      for (const k of ['ent_coef', 'clip_range', 'tau', 'buffer_size', 'train_freq', 'net_arch', 'seed'] as const)
        if (p[k] != null) body[k] = p[k]
      await postJson('/training/start', body)
      set({ telemetry: [], trainingStatus: await getJson('/training/status'), message: `Started ${get().trainingAlgorithm} training.` })
    })
  },

  stopTraining: async () => {
    await get().guard(async () => {
      await postJson('/training/stop', {})
      set({ trainingStatus: await getJson('/training/status'), message: 'Stop requested.' })
    })
  },

  startTuning: async ({ algorithm, nTrials, timestepsPerTrial }) => {
    await get().guard(async () => {
      await postJson('/tuning/start', { algorithm, n_trials: nTrials, timesteps_per_trial: timestepsPerTrial })
      set({ tuningStatus: await getJson('/tuning/status'), message: `Tuning started: ${nTrials} trials of ${algorithm}.` })
    })
  },

  stopTuning: async () => {
    await get().guard(async () => {
      await postJson('/tuning/stop', {})
      set({ tuningStatus: await getJson('/tuning/status'), message: 'Tuning stop requested.' })
    })
  },

  startEvaluation: async (runName, episodes, deterministic) => {
    await get().guard(async () => {
      await postJson('/evaluation/start', { run_name: runName, episodes, deterministic })
      set({ evaluationStatus: await getJson('/evaluation/status'), message: `Evaluating ${runName} (${episodes} episodes).` })
    })
  },

  stopEvaluation: async () => {
    await get().guard(async () => {
      await postJson('/evaluation/stop', {})
      set({ evaluationStatus: await getJson('/evaluation/status'), message: 'Stopping evaluation…' })
    })
  },

  fetchRunDetails: async (name) => {
    try { return await getJson(`/runs/${encodeURIComponent(name)}`) } catch { return null }
  },

  exportRun: async (name) => {
    try { return (await postJson(`/runs/${encodeURIComponent(name)}/export`, {})).path ?? null }
    catch (e: any) { set({ message: String(e?.message || e) }); return null }
  },

  deleteRun: async (name) => {
    try {
      await postJson(`/runs/${encodeURIComponent(name)}/delete`, {})
      await get().loadRuns()
      set({ message: `Deleted run ${name}.` })
      return true
    } catch (e: any) { set({ message: String(e?.message || e) }); return false }
  },

  fetchBackendLogs: async (lines = 500) => {
    try { return ((await getJson(`/logs/backend?lines=${lines}`)).lines ?? []).map(String) } catch { return [] }
  },

  savePreferences: async () => {
    await get().guard(async () => {
      await postJson('/app/preferences', prefsBody(get()))
      set({ appPreferences: await getJson('/app/preferences'), message: 'Saved app preferences.' })
    })
  },

  saveAgentSettings: async (providers) => {
    await get().guard(async () => {
      await postJson('/agent/providers', providers)
      await postJson('/app/preferences', prefsBody(get()))
      set({
        agentSettings: await getJson('/agent/providers'),
        appPreferences: await getJson('/app/preferences'),
        message: 'Saved agent settings.',
      })
    })
    await get().refreshAgentHealth()
  },

  checkModelCapabilities: () => getJson('/agent/capabilities'),

  executeAgentTool: async (name, args) => {
    const res = await postJson('/agents/execute_tool', { name, args })
    const result = res.result ?? {}
    if (result.config) { get().applyConfigResponse(result) }
    get().refreshAll()
    return result
  },

  pollTelemetry: async () => {
    const s = get()
    const trainingActive = s.trainingStatus?.active === true
    const evalActive = s.evaluationStatus?.active === true
    const tuningActive = s.tuningStatus?.active === true
    if (tuningActive) { try { set({ tuningStatus: await getJson('/tuning/status') }) } catch {} }
    if (!trainingActive && !evalActive) return
    try {
      if (trainingActive) {
        const status = await getJson('/training/status')
        const res = await getJson(`/training/telemetry?since=${get().telemetry.length}`)
        const total = res.total ?? 0
        let tele = get().telemetry
        if (total < tele.length) tele = []
        set({ telemetry: [...tele, ...(res.points ?? [])], trainingStatus: status })
        if (status.active !== true) await get().loadRuns()
      }
      if (evalActive) {
        const es = await getJson('/evaluation/status')
        set({ evaluationStatus: es })
        if (es?.active !== true) await get().loadRuns()
      }
    } catch {}
  },

  init: () => {
    if ((globalThis as any).__rtgInit) return
    ;(globalThis as any).__rtgInit = true
    get().refreshAll()
    get().refreshAgentHealth()
    setInterval(() => get().pollTelemetry(), 2000)
    setInterval(() => get().refreshAgentHealth(), 8000)
  },
}))

// --- derived helpers (Flutter getters) ------------------------------------
const clamp = (v: number, lo: number, hi: number) => Math.max(lo, Math.min(hi, v))
const prefsBody = (s: State) => ({
  stream_resolution_scale: s.streamResolutionScale,
  show_inspector_on_dashboard: s.showInspectorOnDashboard,
  agent_autonomy: s.agentAutonomy,
})
const enabledAny = (list: any) => Array.isArray(list) && list.some((x) => x?.enabled === true)

export const agentConnected = (s: State) => s.health?.ok === true && s.agentHealth?.reachable === true
export const currentProjectId = (s: State): string | null => s.envConfig?.project_id ?? null
export const currentProjectName = (s: State) => s.currentProjectPath?.split(/[\\/]/).pop() || null
export const hasRobot = (s: State) => s.robotInfo?.loaded === true || s.robotInfo?.name != null
export const hasEnabledObservations = (s: State) => enabledAny(s.envConfig?.observations)
export const hasEnabledActions = (s: State) => enabledAny(s.envConfig?.actions)
export const hasEnabledRewards = (s: State) => enabledAny(s.envConfig?.rewards)
export const trainingActive = (s: State) => s.trainingStatus?.active === true
export const tuningActive = (s: State) => s.tuningStatus?.active === true
export const evaluationActive = (s: State) => s.evaluationStatus?.active === true
export const anyJobActive = (s: State) => trainingActive(s) || tuningActive(s) || evaluationActive(s)
export const blockingJobLabel = (s: State) => (tuningActive(s) ? 'Tuning' : evaluationActive(s) ? 'Evaluation' : null)

export function trainingBlockers(s: State): string[] {
  const b: string[] = []
  if (!hasRobot(s)) b.push('load a robot')
  if (!hasEnabledObservations(s)) b.push('enable observations')
  if (!hasEnabledActions(s)) b.push('enable actions')
  if (!hasEnabledRewards(s)) b.push('set a reward')
  b.push(...s.configProblems)
  return b
}
export const canStartTraining = (s: State) => trainingBlockers(s).length === 0

export const agentConnectionDetail = (s: State): string => {
  if (s.health?.ok !== true) return 'Backend offline'
  if (!s.agentHealth) return 'Checking provider…'
  const provider = s.agentHealth.provider ?? 'provider'
  if (s.agentHealth.reachable !== true) {
    const d = s.agentHealth.detail
    return `Provider unreachable${d ? ` — ${d}` : ''}`
  }
  const model = s.agentHealth.model ?? 'model'
  if (s.agentHealth.model_available === false) return `Connected (${provider}), but model "${model}" not found`
  return `Connected · ${model}`
}

export { streamPostJson }
