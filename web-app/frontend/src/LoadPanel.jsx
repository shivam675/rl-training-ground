import { useRef, useState } from 'react'
import { loadByPath, uploadModel } from './api.js'

/** UI to add a URDF at runtime — by server path or by uploading files. */
export default function LoadPanel() {
  const [path, setPath] = useState('')
  // Default offset in x so a new model doesn't spawn inside the default R2D2.
  const [base, setBase] = useState(['1.5', '0', '0.6'])
  const [busy, setBusy] = useState(false)
  const [msg, setMsg] = useState(null) // { ok, text }
  const fileRef = useRef(null)

  const baseNums = () => base.map((v) => parseFloat(v) || 0)

  async function run(action) {
    setBusy(true)
    setMsg(null)
    try {
      const { name } = await action()
      setMsg({ ok: true, text: `Loaded "${name}"` })
    } catch (e) {
      setMsg({ ok: false, text: e.message })
    } finally {
      setBusy(false)
    }
  }

  const onLoadPath = () => {
    if (!path.trim()) return setMsg({ ok: false, text: 'Enter a .urdf path' })
    run(() => loadByPath(path.trim(), baseNums()))
  }

  const onUpload = () => {
    const files = fileRef.current?.files
    if (!files || files.length === 0) return setMsg({ ok: false, text: 'Choose files first' })
    run(() => uploadModel(files, baseNums())).then(() => {
      if (fileRef.current) fileRef.current.value = ''
    })
  }

  return (
    <div className="loader">
      <div className="loader__label">Add model</div>

      <div className="loader__row">
        <input
          type="text"
          placeholder="server path or pybullet_data name (e.g. humanoid/humanoid.urdf)"
          value={path}
          onChange={(e) => setPath(e.target.value)}
          onKeyDown={(e) => e.key === 'Enter' && onLoadPath()}
        />
        <button onClick={onLoadPath} disabled={busy}>
          Load path
        </button>
      </div>

      <div className="loader__row">
        <input ref={fileRef} type="file" multiple accept=".urdf,.stl,.obj,.dae,.mtl,.png,.jpg,.jpeg" />
        <button onClick={onUpload} disabled={busy}>
          Upload
        </button>
      </div>

      <div className="loader__row loader__base">
        <span>spawn xyz</span>
        {base.map((v, i) => (
          <input
            key={i}
            type="number"
            step="0.1"
            value={v}
            onChange={(e) => setBase(base.map((b, j) => (j === i ? e.target.value : b)))}
          />
        ))}
      </div>

      {msg && <div className={`loader__msg ${msg.ok ? 'ok' : 'err'}`}>{msg.text}</div>}
      <div className="hint">Upload the .urdf plus its mesh files (or select the whole folder).</div>
    </div>
  )
}
