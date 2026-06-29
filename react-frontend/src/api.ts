// Single typed gateway to the FastAPI backend. Mirrors the Flutter BackendApi:
// errors come back as {detail:{code,message,hint}} — we surface message.

const API = (import.meta.env.VITE_API_URL || 'http://127.0.0.1:8000').replace(/\/$/, '')
export const WS_BASE = API.replace(/^http/, 'ws')
export const WS_SIM = `${WS_BASE}/ws/simulation`
export const WS_AGENT_EVENTS = `${WS_BASE}/ws/agent_events`

export type Json = Record<string, any>

async function unwrap(res: Response): Promise<any> {
  const text = await res.text()
  const data = text ? JSON.parse(text) : {}
  if (!res.ok) {
    const d = data?.detail
    const msg = typeof d === 'string' ? d : d?.message || res.statusText
    const hint = d?.hint ? ` (${d.hint})` : ''
    throw new Error(`${msg}${hint}`)
  }
  return data
}

export function getJson(path: string): Promise<any> {
  return fetch(`${API}${path}`).then(unwrap)
}

export function postJson(path: string, body: Json = {}): Promise<any> {
  return fetch(`${API}${path}`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  }).then(unwrap)
}

// NDJSON stream (one JSON object per line). Yields parsed events as they arrive.
export async function* streamPostJson(
  path: string,
  body: Json,
  signal?: AbortSignal,
): AsyncGenerator<Json> {
  const res = await fetch(`${API}${path}`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
    signal,
  })
  if (!res.body) throw new Error('No response body')
  const reader = res.body.getReader()
  const decoder = new TextDecoder()
  let buf = ''
  while (true) {
    const { value, done } = await reader.read()
    if (done) break
    buf += decoder.decode(value, { stream: true })
    let nl: number
    while ((nl = buf.indexOf('\n')) >= 0) {
      const line = buf.slice(0, nl).trim()
      buf = buf.slice(nl + 1)
      if (line) yield JSON.parse(line)
    }
  }
  const tail = buf.trim()
  if (tail) yield JSON.parse(tail)
}
