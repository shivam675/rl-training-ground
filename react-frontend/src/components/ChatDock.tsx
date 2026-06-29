import { useEffect, useRef, useState } from 'react'
import { useChat, type ChatMessage } from '../chat'
import { useStore, agentConnected, agentConnectionDetail } from '../store'
import { useUi } from '../ui'

export function ChatDock() {
  const collapsed = useUi((s) => s.dockCollapsed)
  const toggle = useUi((s) => s.toggleDock)
  const messages = useChat((s) => s.messages)
  const sending = useChat((s) => s.sending)
  const send = useChat((s) => s.send)
  const clear = useChat((s) => s.clear)
  const transcript = useChat((s) => s.transcript)
  const forceTick = useChat((s) => s.forceScrollTick)
  const connected = useStore(agentConnected)
  const detail = useStore(agentConnectionDetail)
  const [text, setText] = useState('')
  const listRef = useRef<HTMLDivElement>(null)
  const stick = useRef(true)

  // follow bottom unless the user scrolled up
  useEffect(() => {
    const el = listRef.current
    if (el && stick.current) el.scrollTop = el.scrollHeight
  }, [messages])
  useEffect(() => {
    const el = listRef.current
    if (el) { stick.current = true; el.scrollTop = el.scrollHeight }
  }, [forceTick])

  if (collapsed)
    return (
      <div className="dock collapsed">
        <button className="btn ghost" title="Open assistant" onClick={toggle} style={{ margin: 8, height: 40 }}>💬</button>
        <div className="statusdot" style={{ margin: '0 auto', background: connected ? 'var(--success)' : 'var(--danger)' }} />
      </div>
    )

  const submit = () => { const t = text; setText(''); send(t) }
  const showTyping = sending && (messages.length === 0 || messages[messages.length - 1].kind !== 'agent' ||
    (!messages[messages.length - 1].text && !messages[messages.length - 1].thinking))

  return (
    <div className="dock">
      <div className="dock-head">
        <button className="btn ghost sm" title="Collapse" onClick={toggle}>›</button>
        <b className="grow">Assistant</b>
        <span className="statusdot" title={detail} style={{ background: connected ? 'var(--success)' : 'var(--danger)' }} />
        <button className="btn ghost sm" title="Copy transcript" onClick={() => navigator.clipboard?.writeText(transcript())}>⧉</button>
        <button className="btn ghost sm" title="Clear" onClick={clear}>🗑</button>
      </div>

      <div className="dock-msgs" ref={listRef} onScroll={(e) => {
        const el = e.currentTarget
        stick.current = el.scrollHeight - el.scrollTop - el.clientHeight < 60
      }}>
        {messages.map((m) => <Bubble key={m.id} m={m} />)}
        {showTyping && <div className="bubble agent typing"><span /><span /><span /></div>}
      </div>

      <div className="dock-input">
        <textarea
          rows={1} value={text} placeholder="Ask the agent…"
          onChange={(e) => setText(e.target.value)}
          onKeyDown={(e) => { if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); submit() } }}
        />
        <button className="btn primary" disabled={sending || !text.trim()} onClick={submit}>Send</button>
      </div>
    </div>
  )
}

function Bubble({ m }: { m: ChatMessage }) {
  const confirmTool = useChat((s) => s.confirmTool)
  const undoConfig = useChat((s) => s.undoConfig)

  if (m.kind === 'notice') return <div className="bubble notice">{m.text}</div>

  if (m.kind === 'tool') {
    const color = m.toolOk == null ? 'var(--warning)' : m.toolOk ? 'var(--success)' : 'var(--danger)'
    const canUndo = m.toolResult?.config != null && !m.toolResult?.undo_used
    return (
      <div className="bubble tool">
        <div className="row" style={{ gap: 7 }}>
          <span className="statusdot" style={{ background: color }} />
          <b className="mono grow">{m.toolName}</b>
        </div>
        {m.toolArgs && <div className="faint mono" style={{ fontSize: 11, marginTop: 3 }}>{m.toolArgs}</div>}
        <div className="muted" style={{ marginTop: 4 }}>{m.text}</div>
        {m.needsConfirmation && <button className="btn primary sm" style={{ marginTop: 7 }} onClick={() => confirmTool(m.id)}>Run</button>}
        {canUndo && m.toolOk && <button className="btn ghost sm" style={{ marginTop: 7 }} onClick={() => undoConfig(m.id)}>Undo</button>}
      </div>
    )
  }

  return (
    <div className={`bubble ${m.kind}`}>
      {m.thinking && (
        <details className="think">
          <summary>reasoning</summary>
          {m.thinking}
        </details>
      )}
      {m.text}
    </div>
  )
}
