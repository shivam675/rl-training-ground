import { useEffect, useRef, useState } from 'react'

/**
 * Connects to the backend state stream.
 *
 * The high-frequency state (~60 Hz) is written to a ref instead of React state,
 * so it never triggers re-renders — the 3D scene reads it directly in useFrame.
 * Only the scene description (which URDFs to load) and connection status, which
 * change rarely, live in React state.
 */
export function useSimSocket(url) {
  const [scene, setScene] = useState(null)
  const [status, setStatus] = useState('connecting')
  const stateRef = useRef(null)
  const wsRef = useRef(null)

  useEffect(() => {
    let closed = false
    let reconnectTimer = null

    function connect() {
      const ws = new WebSocket(url)
      wsRef.current = ws
      setStatus('connecting')

      ws.onopen = () => setStatus('connected')
      ws.onmessage = (e) => {
        const msg = JSON.parse(e.data)
        if (msg.type === 'scene') setScene(msg)
        else if (msg.type === 'state') stateRef.current = msg
      }
      ws.onclose = () => {
        if (closed) return
        setStatus('disconnected')
        reconnectTimer = setTimeout(connect, 1000) // auto-reconnect in dev
      }
      ws.onerror = () => ws.close()
    }

    connect()
    return () => {
      closed = true
      clearTimeout(reconnectTimer)
      wsRef.current?.close()
    }
  }, [url])

  const send = (obj) => {
    const ws = wsRef.current
    if (ws && ws.readyState === WebSocket.OPEN) ws.send(JSON.stringify(obj))
  }

  return { scene, status, stateRef, send }
}
