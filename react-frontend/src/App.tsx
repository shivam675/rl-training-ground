import { useEffect } from 'react'
import { useStore, agentConnected } from './store'
import { useUi, type Page } from './ui'
import { WS_AGENT_EVENTS } from './api'
import { ChatDock } from './components/ChatDock'
import Home from './panels/Home'
import Robot from './panels/Robot'
import ObsAction from './panels/ObsAction'
import Rewards from './panels/Rewards'
import Training from './panels/Training'
import Evaluation from './panels/Evaluation'
import Settings from './panels/Settings'
import Logs from './panels/Logs'

const NAV: { page: Page; label: string; ico: string }[] = [
  { page: 'home', label: 'Home', ico: '🏠' },
  { page: 'robot', label: 'Robot', ico: '🤖' },
  { page: 'obsaction', label: 'Obs/Act', ico: '🎚️' },
  { page: 'rewards', label: 'Rewards', ico: '🏆' },
  { page: 'training', label: 'Training', ico: '📈' },
  { page: 'evaluation', label: 'Eval', ico: '🎬' },
  { page: 'settings', label: 'Settings', ico: '⚙️' },
  { page: 'logs', label: 'Logs', ico: '📜' },
]

const PANELS: Record<Page, () => JSX.Element> = {
  home: Home, robot: Robot, obsaction: ObsAction, rewards: Rewards,
  training: Training, evaluation: Evaluation, settings: Settings, logs: Logs,
}

export default function App() {
  const init = useStore((s) => s.init)
  const page = useUi((s) => s.page)
  const setPage = useUi((s) => s.setPage)
  const connected = useStore(agentConnected)
  const backendOk = useStore((s) => s.health?.ok === true)
  const message = useStore((s) => s.message)
  const busy = useStore((s) => s.busy)
  const pushToast = useUi((s) => s.pushToast)

  useEffect(() => { init() }, [init])

  // Agent-event stream → toasts (the backend Notifier). Auto-reconnects.
  useEffect(() => {
    let ws: WebSocket | null = null
    let timer: ReturnType<typeof setTimeout> | null = null
    let closed = false
    const connect = () => {
      ws = new WebSocket(WS_AGENT_EVENTS)
      ws.onmessage = (e) => {
        try {
          const ev = JSON.parse(e.data)
          if (ev.type === 'notification' && !ev.replay)
            pushToast({ title: ev.title ?? 'Notice', body: ev.body ?? '', severity: ev.severity ?? 'info' })
        } catch {}
      }
      ws.onclose = () => { if (!closed) timer = setTimeout(connect, 3000) }
      ws.onerror = () => ws?.close()
    }
    connect()
    return () => { closed = true; if (timer) clearTimeout(timer); ws?.close() }
  }, [pushToast])

  const Panel = PANELS[page]

  return (
    <div className="app">
      <nav className="rail">
        <div className="brand">RTG</div>
        {NAV.map((n) => (
          <button key={n.page} className={`rail-btn${page === n.page ? ' active' : ''}`} onClick={() => setPage(n.page)}>
            <span className="ico">{n.ico}</span>{n.label}
          </button>
        ))}
        <div className="spacer" />
        <div className="dot" title={backendOk ? 'Backend online' : 'Backend offline'}
          style={{ background: backendOk ? 'var(--success)' : 'var(--danger)' }} />
      </nav>

      <div className="content">
        <div style={{ flex: 1, minHeight: 0, display: 'flex' }}><Panel /></div>
        <div className="statusbar">
          <span className="statusdot" style={{ background: connected ? 'var(--success)' : backendOk ? 'var(--warning)' : 'var(--danger)' }} />
          <span className="grow" style={{ overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>{message}</span>
          {busy && <span className="faint">working…</span>}
        </div>
      </div>

      <ChatDock />
      <Toasts />
    </div>
  )
}

function Toasts() {
  const toasts = useUi((s) => s.toasts)
  const dismiss = useUi((s) => s.dismissToast)
  const color = (s: string) => s === 'error' ? 'var(--danger)' : s === 'warning' ? 'var(--warning)' : s === 'success' ? 'var(--success)' : 'var(--info)'
  return (
    <div className="toasts">
      {toasts.map((t) => (
        <div key={t.id} className="toast" onClick={() => dismiss(t.id)} style={{ borderLeft: `3px solid ${color(t.severity)}` }}>
          <div className="t-title">{t.title}</div>
          {t.body && <div className="muted">{t.body}</div>}
        </div>
      ))}
    </div>
  )
}
