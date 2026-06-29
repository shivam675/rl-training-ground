import { create } from 'zustand'
import { streamPostJson, type Json } from './api'
import { useStore } from './store'

// Port of the Flutter ChatController: streaming token buffer (flushed every
// 66ms so we don't re-render per token), inline <think> parser, tool-confirm
// flow, and localStorage persistence.

export type ChatKind = 'user' | 'agent' | 'tool' | 'notice'
export interface ChatMessage {
  id: number
  kind: ChatKind
  text: string
  thinking: string
  toolName?: string
  toolArgs?: string
  toolArgsRaw?: Json
  toolResult?: Json
  toolOk?: boolean | null
  needsConfirmation?: boolean
  streaming?: boolean
  time: number
}

const KEY = 'easyrtg.agent_chat.v1'
const FLUSH_MS = 66
let uid = 1
const nid = () => uid++

const welcome = (): ChatMessage => ({
  id: nid(), kind: 'agent', time: 0,
  text: 'Agent ready. I can inspect the robot, test rewards, start or stop training and compare runs for you — just ask.',
  thinking: '',
})

interface ChatState {
  messages: ChatMessage[]
  sending: boolean
  forceScrollTick: number
  send: (text: string) => Promise<void>
  confirmTool: (id: number) => Promise<void>
  undoConfig: (id: number) => Promise<void>
  clear: () => void
  transcript: () => string
}

// Streaming buffers + inline-<think> parser state (module-scoped: one stream
// at a time, mirroring the single-conversation Flutter controller).
let chunkBuf = ''
let thinkBuf = ''
let flushTimer: ReturnType<typeof setTimeout> | null = null
let inThink = false
let tagCarry = ''

export const useChat = create<ChatState>((set, get) => {
  const persist = () => {
    const stable = get().messages.filter((m) => !m.streaming && m.kind !== 'tool')
    try {
      localStorage.setItem(KEY, JSON.stringify(stable.map((m) => ({
        kind: m.kind, text: m.text, thinking: m.thinking || undefined, time: m.time,
      }))))
    } catch {}
  }

  const flush = () => {
    if (flushTimer) { clearTimeout(flushTimer); flushTimer = null }
    if (!chunkBuf && !thinkBuf) return
    const text = chunkBuf, think = thinkBuf
    chunkBuf = ''; thinkBuf = ''
    set((s) => {
      const msgs = s.messages.slice()
      const last = msgs[msgs.length - 1]
      let bubble: ChatMessage
      if (last && last.kind === 'agent' && last.streaming) {
        bubble = { ...last, text: last.text + text, thinking: last.thinking + think }
        msgs[msgs.length - 1] = bubble
      } else {
        bubble = { id: nid(), kind: 'agent', text, thinking: think, streaming: true, time: Date.now() }
        msgs.push(bubble)
      }
      return { messages: msgs }
    })
  }

  const queueChunk = (chunk: string) => {
    const visible = separateThink(chunk)
    if (visible) chunkBuf += visible
    if (!flushTimer) flushTimer = setTimeout(flush, FLUSH_MS)
  }
  const queueThinking = (chunk: string) => {
    if (!chunk) return
    thinkBuf += chunk
    if (!flushTimer) flushTimer = setTimeout(flush, FLUSH_MS)
  }

  // Split inline <think>...</think> out of the content stream into thinkBuf.
  const separateThink = (incoming: string): string => {
    let visible = ''
    let s = tagCarry + incoming
    tagCarry = ''
    while (s) {
      if (!inThink) {
        const open = s.indexOf('<think>')
        if (open < 0) { const keep = partialTail(s, '<think>'); visible += s.slice(0, s.length - keep); tagCarry = s.slice(s.length - keep); break }
        visible += s.slice(0, open); s = s.slice(open + 7); inThink = true
      } else {
        const close = s.indexOf('</think>')
        if (close < 0) { const keep = partialTail(s, '</think>'); thinkBuf += s.slice(0, s.length - keep); tagCarry = s.slice(s.length - keep); break }
        thinkBuf += s.slice(0, close); s = s.slice(close + 8); inThink = false
      }
    }
    return visible
  }

  const finishBubble = () => {
    flush()
    inThink = false; tagCarry = ''
    set((s) => ({ messages: s.messages.map((m) => (m.streaming ? { ...m, streaming: false } : m)) }))
    persist()
  }

  const add = (m: Omit<ChatMessage, 'id' | 'time'>) =>
    set((s) => ({ messages: [...s.messages, { ...m, id: nid(), time: Date.now() } as ChatMessage] }))

  return {
    messages: loadHistory(),
    sending: false,
    forceScrollTick: 0,

    clear: () => { set({ messages: [welcome()] }); persist() },
    transcript: () => get().messages.map(transcriptLine).join('\n\n'),

    send: async (raw) => {
      const text = raw.trim()
      if (!text || get().sending) return
      add({ kind: 'user', text, thinking: '' })
      set((s) => ({ sending: true, forceScrollTick: s.forceScrollTick + 1 }))
      persist()
      const history = historyForRequest(get().messages)
      try {
        for await (const ev of streamPostJson('/agents/chat/stream', { agent: 'helper', message: text, context: {}, history })) {
          if (ev.type === 'error') throw new Error(ev.detail || 'Agent stream failed.')
          switch (ev.type) {
            case 'chunk': queueChunk(ev.text ?? ''); break
            case 'thinking': queueThinking(ev.text ?? ''); break
            case 'tool_call':
              finishBubble()
              add({
                kind: 'tool', text: 'running…', thinking: '',
                toolName: ev.name ?? 'tool', toolArgs: argsSummary(ev.args),
                toolArgsRaw: ev.args && typeof ev.args === 'object' ? ev.args : undefined,
                toolOk: null,
              })
              break
            case 'tool_result': {
              const result = ev.result
              const needsConfirm = result && typeof result === 'object' && result.requires_confirmation === true
              const error = result && typeof result === 'object' ? result.error : null
              set((s) => {
                const msgs = s.messages.slice()
                let i = msgs.map((m) => m.kind === 'tool' && m.toolOk == null).lastIndexOf(true)
                if (i < 0) i = msgs.length - 1
                const t = { ...msgs[i] }
                if (needsConfirm) { t.needsConfirmation = true; t.text = 'awaiting your confirmation' }
                else { t.toolOk = error == null; t.text = error ?? 'done' }
                if (result && typeof result === 'object') t.toolResult = result
                msgs[i] = t
                return { messages: msgs }
              })
              break
            }
            case 'notice':
              finishBubble()
              add({ kind: 'notice', text: ev.text ?? '', thinking: '' })
              break
            case 'done':
              useStore.getState().refreshAll()
              break
          }
        }
        finishBubble()
        set({ sending: false })
        persist()
      } catch (e: any) {
        finishBubble()
        add({ kind: 'agent', text: `⚠ ${String(e?.message || e).replace('Error: ', '')}`, thinking: '' })
        set({ sending: false })
        persist()
      }
    },

    confirmTool: async (id) => {
      set((s) => ({ messages: s.messages.map((m) => (m.id === id ? { ...m, needsConfirmation: false, toolOk: null, text: 'running…' } : m)) }))
      const tool = get().messages.find((m) => m.id === id)
      if (!tool) return
      try {
        const result = await useStore.getState().executeAgentTool(tool.toolName ?? '', tool.toolArgsRaw ?? {})
        const error = result.error
        set((s) => ({ messages: s.messages.map((m) => (m.id === id ? { ...m, toolResult: result, toolOk: error == null, text: error ?? 'done' } : m)) }))
      } catch (e: any) {
        set((s) => ({ messages: s.messages.map((m) => (m.id === id ? { ...m, toolOk: false, text: String(e?.message || e) } : m)) }))
      }
    },

    undoConfig: async (sourceId) => {
      add({ kind: 'tool', text: 'running...', thinking: '', toolName: 'undo_config_change', toolOk: null })
      const undoId = get().messages[get().messages.length - 1].id
      const result = await useStore.getState().undoConfigChange()
      const error = result.error
      set((s) => ({
        messages: s.messages.map((m) =>
          m.id === undoId ? { ...m, toolResult: result, toolOk: error == null, text: error ?? 'done' }
          : m.id === sourceId ? { ...m, toolResult: { ...m.toolResult, undo_used: error == null } }
          : m),
      }))
    },
  }
})

function partialTail(s: string, tag: string): number {
  const max = Math.min(s.length, tag.length - 1)
  for (let len = max; len > 0; len--) if (tag.startsWith(s.slice(s.length - len))) return len
  return 0
}
function argsSummary(args: any): string {
  if (!args || typeof args !== 'object' || !Object.keys(args).length) return ''
  return Object.entries(args).map(([k, v]) => `${k}: ${v}`).join(', ')
}
function transcriptLine(m: ChatMessage): string {
  switch (m.kind) {
    case 'user': return `You: ${m.text}`
    case 'agent': return `Agent: ${m.text}`
    case 'tool': return `[tool] ${m.toolName}(${m.toolArgs ?? ''}) → ${m.text}`
    case 'notice': return `[notice] ${m.text}`
  }
}
function historyForRequest(messages: ChatMessage[]) {
  const turns: { role: string; content: string }[] = []
  for (const m of messages) {
    if (m.kind === 'user') turns.push({ role: 'user', content: m.text })
    else if (m.kind === 'agent' && m.text) turns.push({ role: 'assistant', content: m.text })
  }
  return turns.slice(Math.max(0, turns.length - 12))
}
function loadHistory(): ChatMessage[] {
  try {
    const raw = localStorage.getItem(KEY)
    if (!raw) return [welcome()]
    const arr = JSON.parse(raw)
    if (!Array.isArray(arr)) return [welcome()]
    const msgs = arr
      .filter((d: any) => d && d.kind !== 'tool' && (d.text || d.thinking))
      .map((d: any): ChatMessage => ({ id: nid(), kind: d.kind, text: d.text ?? '', thinking: d.thinking ?? '', time: d.time ?? 0 }))
    return msgs.length ? msgs : [welcome()]
  } catch { return [welcome()] }
}
