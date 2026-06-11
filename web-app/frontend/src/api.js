// REST helpers for loading URDFs at runtime. The loaded model shows up in the
// 3D view automatically when the backend broadcasts the updated `scene` message.

async function unwrap(res) {
  let body = {}
  try {
    body = await res.json()
  } catch {
    /* non-JSON error */
  }
  if (!res.ok || body.ok === false) {
    throw new Error(body.error || `${res.status} ${res.statusText}`)
  }
  return body
}

export async function loadByPath(path, base) {
  const res = await fetch('/api/load_path', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ path, base }),
  })
  return unwrap(res)
}

export async function uploadModel(fileList, base) {
  const fd = new FormData()
  for (const f of fileList) {
    // Preserve relative folder structure when a directory was selected, so the
    // backend can resolve meshes referenced like "meshes/foo.stl".
    fd.append('files', f, f.webkitRelativePath || f.name)
  }
  fd.append('base', JSON.stringify(base))
  const res = await fetch('/api/upload', { method: 'POST', body: fd })
  return unwrap(res)
}
