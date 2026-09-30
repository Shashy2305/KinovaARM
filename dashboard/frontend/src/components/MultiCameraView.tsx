import { useState } from 'react'
import { cameraStreamUrl } from '../lib/api'

const CAMERAS = [
  { id: 'oakd', label: 'OAK-D' },
  { id: 'realsense', label: 'RealSense' },
  { id: 'wrist', label: 'Wrist' },
] as const

export function MultiCameraView() {
  const [reloadKey, setReloadKey] = useState(0)

  return (
    <div className="panel overflow-hidden">
      <div className="panel-header">
        <h2 className="font-semibold text-sm tracking-tight">Multi-Camera View</h2>
        <button className="btn" onClick={() => setReloadKey((k) => k + 1)}>
          ↻ Reload streams
        </button>
      </div>
      <div className="grid grid-cols-1 md:grid-cols-3 gap-px bg-(--color-border)">
        {CAMERAS.map((cam) => (
          <div key={cam.id} className="bg-(--color-panel) relative aspect-video">
            <img
              key={`${cam.id}-${reloadKey}`}
              src={cameraStreamUrl(cam.id)}
              alt={cam.label}
              className="w-full h-full object-cover"
              onError={(e) => {
                ;(e.target as HTMLImageElement).style.opacity = '0.15'
              }}
            />
            <div className="absolute top-0 left-0 right-0 flex items-center justify-between px-2 py-1 bg-gradient-to-b from-black/70 to-transparent">
              <span className="text-xs font-mono font-semibold">{cam.label}</span>
              <span className="live-dot text-(--color-red)" style={{ background: 'currentColor' }} />
            </div>
          </div>
        ))}
      </div>
    </div>
  )
}
