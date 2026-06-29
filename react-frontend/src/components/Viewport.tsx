import { useEffect, useRef, useState } from 'react'
import { WS_SIM } from '../api'
import { useStore } from '../store'

// Live PyBullet viewport. Binary frames are JPEG or RTGF-raw (magic "RTGF" +
// uint32-LE w + uint32-LE h + RGBA8888). Decode happens off the main thread via
// createImageBitmap and draws straight to <canvas> — React never re-renders per
// frame, only on the ~1Hz status text.

interface Status { connected: boolean; training: boolean; text: string; mode: string | null; paused: boolean }
const isRtgf = (b: Uint8Array) => b.length > 12 && b[0] === 0x52 && b[1] === 0x54 && b[2] === 0x47 && b[3] === 0x46

export function Viewport() {
  const wrapRef = useRef<HTMLDivElement>(null)
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const wsRef = useRef<WebSocket | null>(null)
  const decodeBusy = useRef(false)
  const lastSize = useRef({ w: 0, h: 0 })
  const drag = useRef<{ x: number; y: number; buttons: number } | null>(null)
  const scale = useStore((s) => s.streamResolutionScale)
  const scaleRef = useRef(scale)
  scaleRef.current = scale
  const [status, setStatus] = useState<Status>({ connected: false, training: false, text: 'connecting', mode: null, paused: false })

  const send = (data: object) => {
    const ws = wsRef.current
    if (ws?.readyState === WebSocket.OPEN) try { ws.send(JSON.stringify(data)) } catch {}
  }

  const draw = (bmp: ImageBitmap) => {
    const c = canvasRef.current
    if (!c) { bmp.close(); return }
    const ctx = c.getContext('2d')!
    const cw = c.width, ch = c.height
    const s = Math.min(cw / bmp.width, ch / bmp.height)
    const dw = bmp.width * s, dh = bmp.height * s
    ctx.fillStyle = '#000'
    ctx.fillRect(0, 0, cw, ch)
    ctx.drawImage(bmp, (cw - dw) / 2, (ch - dh) / 2, dw, dh)
    bmp.close()
  }

  const onFrame = async (buf: ArrayBuffer) => {
    if (decodeBusy.current) return // drop frames while a decode is in flight
    decodeBusy.current = true
    setStatus((p) => (p.connected && !p.training ? p : { ...p, connected: true, training: false }))
    try {
      const bytes = new Uint8Array(buf)
      let bmp: ImageBitmap
      if (isRtgf(bytes)) {
        const dv = new DataView(buf)
        const w = dv.getUint32(4, true), h = dv.getUint32(8, true)
        const px = new Uint8ClampedArray(buf, 12, w * h * 4)
        bmp = await createImageBitmap(new ImageData(px, w, h))
      } else {
        bmp = await createImageBitmap(new Blob([buf]))
      }
      draw(bmp)
    } catch {} finally { decodeBusy.current = false }
  }

  // connect + reconnect
  useEffect(() => {
    let closed = false
    let timer: ReturnType<typeof setTimeout> | null = null
    const connect = () => {
      const ws = new WebSocket(WS_SIM)
      ws.binaryType = 'arraybuffer'
      wsRef.current = ws
      ws.onmessage = (e) => {
        if (typeof e.data === 'string') {
          try {
            const d = JSON.parse(e.data)
            const vp = d.viewport
            const label = vp && vp.render_width > 0
              ? `${Math.round(vp.render_width)}x${Math.round(vp.render_height)}${vp.limit != null ? ' CPU cap' : ''}` : null
            setStatus({
              connected: true,
              training: d.training === true,
              text: d.training === true ? 'Training…' : [d.renderer ?? 'stream', `${Math.round(d.fps ?? 0)} fps`, label].filter(Boolean).join(' | '),
              mode: d.mode ?? null,
              paused: d.paused === true,
            })
          } catch {}
        } else {
          onFrame(e.data as ArrayBuffer)
        }
      }
      ws.onclose = () => { wsRef.current = null; if (!closed) { setStatus((p) => ({ ...p, connected: false, text: 'disconnected' })); timer = setTimeout(connect, 2000) } }
      ws.onerror = () => ws.close()
    }
    connect()
    return () => { closed = true; if (timer) clearTimeout(timer); wsRef.current?.close() }
  }, [])

  // size canvas to container * dpr * scale, push resize to backend
  useEffect(() => {
    const wrap = wrapRef.current!, canvas = canvasRef.current!
    const ro = new ResizeObserver(() => {
      const dpr = window.devicePixelRatio || 1
      // canvas backing always tracks the container (independent of the throttled
      // resolution-scale render request below)
      const cw = Math.round(wrap.clientWidth * dpr), ch = Math.round(wrap.clientHeight * dpr)
      if (canvas.width !== cw || canvas.height !== ch) { canvas.width = cw; canvas.height = ch }
      const w = Math.round(cw * scaleRef.current), h = Math.round(ch * scaleRef.current)
      if (Math.abs(w - lastSize.current.w) <= 8 && Math.abs(h - lastSize.current.h) <= 8) return
      lastSize.current = { w, h }
      send({ cmd: 'resize', width: w, height: h, scale: scaleRef.current })
    })
    ro.observe(wrap)
    return () => ro.disconnect()
  }, [])
  // scale changed → re-request at the new resolution immediately
  useEffect(() => {
    const wrap = wrapRef.current
    if (!wrap) return
    const dpr = window.devicePixelRatio || 1
    const w = Math.round(wrap.clientWidth * dpr * scale)
    const h = Math.round(wrap.clientHeight * dpr * scale)
    lastSize.current = { w, h }
    send({ cmd: 'resize', width: w, height: h, scale })
  }, [scale])

  return (
    <div
      className="viewport" ref={wrapRef}
      style={{ cursor: 'move' }}
      onPointerDown={(e) => { (e.target as Element).setPointerCapture(e.pointerId); drag.current = { x: e.clientX, y: e.clientY, buttons: e.buttons } }}
      onPointerUp={() => (drag.current = null)}
      onPointerMove={(e) => {
        const d = drag.current
        if (!d) return
        const dx = e.clientX - d.x, dy = e.clientY - d.y
        d.x = e.clientX; d.y = e.clientY
        const pan = (e.buttons & 2) !== 0 || (e.buttons & 4) !== 0
        send({ cmd: pan ? 'pan' : 'orbit', dx, dy })
      }}
      onWheel={(e) => send({ cmd: 'zoom', notches: -e.deltaY / 120 })}
      onContextMenu={(e) => e.preventDefault()}
    >
      <canvas ref={canvasRef} />

      {status.training && (
        <div className="vp-overlay">
          <div className="spinner" />
          <b style={{ color: '#fff' }}>Training in progress</b>
          <div style={{ maxWidth: 320, fontSize: 12.5 }}>Live view is paused so the simulator runs at full speed. Watch the reward curve in Training.</div>
        </div>
      )}
      {!status.connected && !status.training && (
        <div className="vp-overlay">
          <div>📡</div>
          <b>{status.text === 'disconnected' ? 'Stream disconnected' : 'Waiting for frame stream…'}</b>
          <div className="faint" style={{ fontSize: 12 }}>{status.text === 'disconnected' ? 'Retrying every 2s.' : WS_SIM}</div>
        </div>
      )}

      <div style={{ position: 'absolute', left: 12, top: 12 }}>
        <div className="glass row" style={{ gap: 7 }}>
          {status.mode ? <span style={{ color: 'var(--warning)', fontWeight: 600 }}>▶ {status.mode}</span> : (
            <>
              <span className="statusdot" style={{ background: status.connected ? 'var(--success)' : 'var(--danger)' }} />
              <span>{status.text}</span>
            </>
          )}
        </div>
      </div>

      {!status.training && (
        <div style={{ position: 'absolute', right: 10, top: 10 }}>
          <div className="glass row" style={{ gap: 4, padding: 4 }}>
            <button className="btn ghost sm" title="Pause/Resume" onClick={() => send({ cmd: 'pause' })}>⏯</button>
            <button className="btn ghost sm" title="Step" onClick={() => send({ cmd: 'step' })}>⏭</button>
            <button className="btn ghost sm" title="Reset" onClick={() => send({ cmd: 'reset' })}>↺</button>
          </div>
        </div>
      )}
      {!status.training && (
        <div style={{ position: 'absolute', left: 12, bottom: 10 }}>
          <div className="glass" style={{ fontSize: 11, opacity: 0.8 }}>Drag: orbit · Right/middle drag: pan · Scroll: zoom</div>
        </div>
      )}
    </div>
  )
}
