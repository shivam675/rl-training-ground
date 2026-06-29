import { useEffect, useState } from 'react'
import { useStore } from '../store'
import { useUi, ACCENT_NAMES } from '../ui'

export default function Settings() {
  const theme = useUi((s) => s.theme)
  const setTheme = useUi((s) => s.setTheme)
  const accent = useUi((s) => s.accent)
  const setAccent = useUi((s) => s.setAccent)

  const agentSettings = useStore((s) => s.agentSettings)
  const scale = useStore((s) => s.streamResolutionScale)
  const autonomy = useStore((s) => s.agentAutonomy)
  const showInspector = useStore((s) => s.showInspectorOnDashboard)
  const busy = useStore((s) => s.busy)
  const set = useStore((s) => s.set)
  const savePrefs = useStore((s) => s.savePreferences)
  const saveAgent = useStore((s) => s.saveAgentSettings)
  const checkCaps = useStore((s) => s.checkModelCapabilities)

  const [providers, setProviders] = useState<any>(null)
  const [caps, setCaps] = useState<any>(null)
  useEffect(() => { if (agentSettings && !providers) setProviders(structuredClone(agentSettings)) }, [agentSettings, providers])

  if (!providers) return <div className="scroll"><div className="muted">Loading settings…</div></div>
  const active: string = providers.active_provider ?? 'ollama'
  const setField = (prov: string, key: string, val: any) => setProviders((p: any) => ({ ...p, [prov]: { ...p[prov], [key]: val } }))

  const OLLAMA = [
    ['base_url', 'Base URL', 'text'], ['model_name', 'Model', 'text'], ['bearer_token', 'Bearer token', 'password'],
    ['temperature', 'Temperature', 'number'], ['top_p', 'top_p', 'number'], ['num_predict', 'num_predict', 'number'],
    ['timeout_seconds', 'Timeout (s)', 'number'],
  ] as const
  const OPENAI = [
    ['base_url', 'Base URL', 'text'], ['model_name', 'Model', 'text'], ['api_key', 'API key', 'password'],
    ['temperature', 'Temperature', 'number'], ['top_p', 'top_p', 'number'], ['max_tokens', 'max_tokens', 'number'],
    ['timeout_seconds', 'Timeout (s)', 'number'], ['reasoning_budget', 'Reasoning budget', 'number'],
  ] as const

  const fields = (prov: string, defs: readonly (readonly [string, string, string])[]) => (
    <div className="grid2">
      {defs.map(([k, label, type]) => (
        <label key={k} className="field">{label}
          <input type={type} value={providers[prov][k] ?? ''} onChange={(e) => setField(prov, k, type === 'number' ? +e.target.value : e.target.value)} />
        </label>
      ))}
      <label className="check"><input type="checkbox" checked={!!providers[prov].enable_thinking} onChange={(e) => setField(prov, 'enable_thinking', e.target.checked)} />Enable thinking</label>
    </div>
  )

  return (
    <div className="scroll">
      <div className="page-title">Settings</div>
      <div className="page-sub">Theme, agent provider and viewport.</div>

      <div className="card">
        <div className="card-head"><span className="ico">🎨</span>Appearance</div>
        <div className="card-body col">
          <div className="row">
            <span className="muted grow">Theme</span>
            <button className={`btn sm ${theme === 'dark' ? 'primary' : ''}`} onClick={() => setTheme('dark')}>Dark</button>
            <button className={`btn sm ${theme === 'light' ? 'primary' : ''}`} onClick={() => setTheme('light')}>Light</button>
          </div>
          <div className="row wrap" style={{ gap: 8 }}>
            <span className="muted">Accent</span>
            {ACCENT_NAMES.map((a) => (
              <button key={a} title={a} onClick={() => setAccent(a)} style={{
                width: 26, height: 26, borderRadius: '50%', cursor: 'pointer',
                border: accent === a ? '2px solid var(--text)' : '2px solid transparent',
                background: swatch(a),
              }} />
            ))}
          </div>
        </div>
      </div>

      <div className="card">
        <div className="card-head"><span className="ico">🤖</span>Assistant provider</div>
        <div className="card-body col">
          <div className="row">
            <label className="check" style={{ padding: 0 }}><input type="radio" checked={active === 'ollama'} onChange={() => setProviders((p: any) => ({ ...p, active_provider: 'ollama' }))} />Ollama</label>
            <label className="check" style={{ padding: 0 }}><input type="radio" checked={active === 'openai'} onChange={() => setProviders((p: any) => ({ ...p, active_provider: 'openai' }))} />OpenAI-compatible</label>
          </div>
          {active === 'ollama' ? fields('ollama', OLLAMA) : fields('openai', OPENAI)}
          <label className="field">System prompt override
            <textarea rows={2} value={providers[active].system_prompt_override ?? ''} onChange={(e) => setField(active, 'system_prompt_override', e.target.value)} />
          </label>
          <div className="row">
            <button className="btn primary" disabled={busy} onClick={() => saveAgent(providers)}>Save provider</button>
            <button className="btn" disabled={busy} onClick={async () => setCaps(await checkCaps())}>Check capabilities</button>
            {caps && <span className="tag">{caps.supports_tools ? '✓ tools' : 'no tool support'}{caps.model ? ` · ${caps.model}` : ''}</span>}
          </div>
        </div>
      </div>

      <div className="card">
        <div className="card-head"><span className="ico">🎚️</span>Behaviour</div>
        <div className="card-body col">
          <label className="field">Viewport resolution scale — {scale.toFixed(2)}×
            <input type="range" min={0.5} max={1.5} step={0.05} value={scale}
              onChange={(e) => set({ streamResolutionScale: +e.target.value })} onMouseUp={() => savePrefs()} onTouchEnd={() => savePrefs()} />
          </label>
          <div className="row">
            <span className="muted grow">Agent autonomy</span>
            <button className={`btn sm ${autonomy === 'act' ? 'primary' : ''}`} onClick={() => { set({ agentAutonomy: 'act' }); savePrefs() }}>Act freely</button>
            <button className={`btn sm ${autonomy === 'ask' ? 'primary' : ''}`} onClick={() => { set({ agentAutonomy: 'ask' }); savePrefs() }}>Ask first</button>
          </div>
          <label className="check"><input type="checkbox" checked={showInspector} onChange={(e) => { set({ showInspectorOnDashboard: e.target.checked }); savePrefs() }} />Show inspector on dashboard</label>
        </div>
      </div>
    </div>
  )
}

// each accent's ring uses its real color (CSS var only exposes the active one)
const COLORS: Record<string, string> = {
  teal: '#4fb7a8', ocean: '#4f9cff', violet: '#9a7bff', magenta: '#e060b8',
  crimson: '#ef5466', amber: '#e8a33d', lime: '#8fc742', slate: '#8a9bb0',
}
const swatch = (a: string) => COLORS[a] ?? '#888'
