import { useRef, useState } from 'react'
import { api, cameraStreamUrl } from '../lib/api'

interface Marker { x: number; y: number }

const CAMERAS = [
  { id: 'wrist', label: 'Wrist', note: 'Best accuracy — jog the arm up close to each corner before clicking.' },
  { id: 'oakd', label: 'OAK-D', note: 'Usually far from the table — clicks here are the least precise.' },
  { id: 'realsense', label: 'RealSense', note: 'Fixed-mount, moderate distance.' },
  { id: 'realsense2', label: 'RealSense 2', note: 'Second fixed-mount camera — good if it has a closer or less occluded angle than the first.' },
] as const

type CameraId = (typeof CAMERAS)[number]['id']

export function WorkspaceBoundary() {
  const imgRef = useRef<HTMLImageElement>(null)
  const [camera, setCamera] = useState<CameraId>('wrist')
  const [markers, setMarkers] = useState<Marker[]>([])
  const [status, setStatus] = useState<string | null>(null)
  const [numPoints, setNumPoints] = useState(0)
  const [saving, setSaving] = useState(false)

  function switchCamera(next: CameraId) {
    setCamera(next)
    setMarkers([])
    setNumPoints(0)
    setStatus(`switched to ${next} — points from other cameras don't carry over, this is a fresh session`)
  }

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
    api.boundaryClick(camera, u, v).then((res) => {
      setStatus(res.status)
      setNumPoints(res.num_points)
    }).catch((err) => setStatus(String(err)))
  }

  async function reset() {
    await api.boundaryReset(camera)
    setMarkers([])
    setNumPoints(0)
    setStatus('reset')
  }

  async function save() {
    setSaving(true)
    try {
      const res = await api.boundarySave(camera)
      setStatus(res.status)
    } finally {
      setSaving(false)
    }
  }

  const activeCam = CAMERAS.find((c) => c.id === camera)!
  const rejected = status?.toLowerCase().includes('not saving')

  return (
    <div className="panel">
      <div className="panel-header">
        <h2 className="font-semibold text-sm tracking-tight">Workspace Boundary</h2>
        <div className="flex gap-2">
          <button className="btn" onClick={reset}>Reset</button>
          <button className="btn btn-primary" disabled={numPoints < 2 || saving} onClick={save}>
            {saving ? 'Saving…' : `Save (${numPoints} pts)`}
          </button>
        </div>
      </div>
      <div className="p-5">
        <div className="flex gap-2 mb-3">
          {CAMERAS.map((c) => (
            <button
              key={c.id}
              onClick={() => switchCamera(c.id)}
              className={`px-3.5 py-1.5 rounded-full text-sm font-medium transition-all ${
                camera === c.id
                  ? 'bg-gradient-to-r from-(--color-brand-from) to-(--color-brand-to) text-white'
                  : 'bg-white/[0.04] text-(--color-text-dim) hover:text-(--color-text)'
              }`}
            >
              {c.label}
            </button>
          ))}
        </div>
        <p className="text-xs text-(--color-text-dim) mb-4">{activeCam.note}</p>

        <div className="relative rounded-xl overflow-hidden border border-(--color-border) cursor-crosshair select-none">
          <img
            ref={imgRef}
            src={cameraStreamUrl(camera)}
            alt={activeCam.label}
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
        {status && (
          <div className={`text-xs font-mono mt-2 ${rejected ? 'text-(--color-red)' : 'text-(--color-text-dim)'}`}>
            {status}
          </div>
        )}
      </div>
    </div>
  )
}
