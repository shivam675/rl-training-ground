import { useEffect, useState } from 'react'
import { useSimSocket } from './useSimSocket.js'
import Viewer from './Viewer.jsx'
import LoadPanel from './LoadPanel.jsx'

const WS_URL = `ws://${location.host}/ws`

export default function App() {
  const { scene, status, stateRef, send } = useSimSocket(WS_URL)

  return (
    <div className="app">
      <Viewer scene={scene} stateRef={stateRef} />
      <div className="hud">
        <div className="title">RL Training Ground</div>
        <div className={`badge badge--${status}`}>{status}</div>
        <div className="buttons">
          <button onClick={() => send({ cmd: 'pause' })}>Pause / Resume</button>
          <button onClick={() => send({ cmd: 'reset' })}>Reset</button>
        </div>
        <SimClock stateRef={stateRef} />
        <div className="hint">drag: orbit · right-drag: pan · wheel: zoom</div>
        <LoadPanel />
      </div>
    </div>
  )
}

/** Lightweight readout: polls the state ref a few times a second (no render storm). */
function SimClock({ stateRef }) {
  const [info, setInfo] = useState({ t: 0, rate: 0 })
  useEffect(() => {
    let last = null
    let lastT = 0
    const id = setInterval(() => {
      const st = stateRef.current
      if (!st) return
      const now = performance.now()
      let rate = info.rate
      if (last !== null) {
        const dt = (now - last) / 1000
        if (dt > 0) rate = 0.7 * info.rate + 0.3 * ((st.t - lastT) / dt)
      }
      last = now
      lastT = st.t
      setInfo({ t: st.t, rate })
    }, 500)
    return () => clearInterval(id)
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [stateRef])

  return (
    <div className="clock">
      sim&nbsp;t&nbsp;=&nbsp;{info.t.toFixed(2)}s
    </div>
  )
}
