import { useRef, useState } from 'react'
import { api, cameraStreamUrl } from '../lib/api'

interface Marker { x: number; y: number }

export function WorkspaceBoundary() {
  const imgRef = useRef<HTMLImageElement>(null)
  const [markers, setMarkers] = useState<Marker[]>([])
  const [status, setStatus] = useState<string | null>(null)
  const [numPoints, setNumPoints] = useState(0)

  // Click coordinates must be translated from the DISPLAYED (CSS-scaled)
  // image size to the camera's NATIVE pixel size before going to the
  // backend -- a mismatch here is exactly the bug that made
  // calibration/define_workspace_boundary.py's own clicks land on the
  // wrong pixel earlier in development. object-fit:cover means the
  // rendered image may also be cropped relative to its natural size, not
  // just scaled, so account for that too.
  function handleClick(e: React.MouseEvent<HTMLImageElement>) {
    const img = imgRef.current
    if (!img || !img.naturalWidth) return
    const rect = img.getBoundingClientRect()
    const scale = Math.max(rect.width / img.naturalWidth, rect.height / img.naturalHeight)
    const renderedW = img.naturalWidth * scale
    const renderedH = img.naturalHeight * scale
    const offsetX = (renderedW - rect.width) / 2
    const offsetY = (renderedH - rect.height) / 2

    const xInRendered = (e.clientX - rect.left) + offsetX
    const yInRendered = (e.clientY - rect.top) + offsetY
    const u = Math.round(xInRendered / scale)
    const v = Math.round(yInRendered / scale)

    setMarkers((m) => [...m, { x: e.clientX - rect.left, y: e.clientY - rect.top }])
    api.boundaryClick('oakd', u, v).then((res) => {
      setStatus(res.status)
      setNumPoints(res.num_points)
    }).catch((err) => setStatus(String(err)))
  }

  async function reset() {
    await api.boundaryReset()
    setMarkers([])
    setNumPoints(0)
    setStatus('reset')
  }

  async function save() {
    const res = await api.boundarySave()
    setStatus(res.status)
  }

  return (
    <div className="panel">
      <div className="panel-header">
        <h2 className="font-semibold text-sm tracking-tight">Workspace Boundary</h2>
        <div className="flex gap-2">
          <button className="btn" onClick={reset}>Reset</button>
          <button className="btn btn-primary" disabled={numPoints < 2} onClick={save}>
            Save ({numPoints} pts)
          </button>
        </div>
      </div>
      <div className="p-4">
        <p className="text-xs text-(--color-text-dim) mb-3">
          Click the corners of the area the arm can actually reach — not just the visible table
          edges. At least 2 points (opposite corners).
        </p>
        <div className="relative rounded-lg overflow-hidden border border-(--color-border) cursor-crosshair select-none">
          <img
            ref={imgRef}
            src={cameraStreamUrl('oakd')}
            alt="OAK-D"
            className="w-full aspect-video object-cover block"
            onClick={handleClick}
          />
          {markers.map((m, i) => (
            <div
              key={i}
              className="absolute w-4 h-4 -translate-x-1/2 -translate-y-1/2 pointer-events-none"
              style={{ left: m.x, top: m.y }}
            >
              <div className="w-full h-full rounded-full border-2 border-(--color-red) bg-(--color-red)/30" />
              <span className="absolute -top-4 left-1/2 -translate-x-1/2 text-[10px] font-mono text-(--color-red)">
                {i + 1}
              </span>
            </div>
          ))}
        </div>
        {status && <div className="text-xs font-mono mt-2 text-(--color-text-dim)">{status}</div>}
      </div>
    </div>
  )
}
