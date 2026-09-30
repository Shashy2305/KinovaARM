import { useEffect, useState } from 'react'
import { api, boardPngUrl, cameraStreamUrl } from '../lib/api'
import type { CaptureResult } from '../lib/types'

const TARGETS = [
  { id: 'oakd', label: 'OAK-D' },
  { id: 'realsense', label: 'RealSense' },
] as const

function CameraCapturePanel({ camera, label }: { camera: string; label: string }) {
  const [visible, setVisible] = useState<{ wrist_ready: boolean; camera_ready: boolean; board_visible: boolean } | null>(null)
  const [capturing, setCapturing] = useState(false)
  const [result, setResult] = useState<CaptureResult | null>(null)
  const [message, setMessage] = useState<string | null>(null)

  useEffect(() => {
    let stop = false
    async function poll() {
      try {
        const v = await api.boardVisible(camera)
        if (!stop) setVisible(v)
      } catch {
        // backend not reachable this tick -- keep last known state, try again
      }
      if (!stop) setTimeout(poll, 800)
    }
    poll()
    return () => { stop = true }
  }, [camera])

  async function doCapture() {
    setCapturing(true)
    setMessage(null)
    try {
      const res = await api.capture(camera)
      setMessage(res.message)
      if (res.ok && res.result) setResult(res.result)
    } catch (e) {
      setMessage(String(e))
    } finally {
      setCapturing(false)
    }
  }

  const ready = visible?.board_visible ?? false

  return (
    <div className="rounded-lg border border-(--color-border) overflow-hidden">
      <div className="relative aspect-video bg-black">
        <img src={cameraStreamUrl(camera)} alt={label} className="w-full h-full object-cover" />
        <div
          className={`absolute inset-0 border-4 transition-colors pointer-events-none ${
            ready ? 'border-(--color-green)' : 'border-transparent'
          }`}
        />
      </div>
      <div className="p-3 bg-(--color-panel-raised)">
        <div className="flex items-center justify-between mb-2">
          <span className="font-medium text-sm">{label}</span>
          <span className={`text-xs font-mono ${ready ? 'text-(--color-green)' : 'text-(--color-text-dim)'}`}>
            {visible === null ? '…' : ready ? 'BOARD DETECTED' : 'no board'}
          </span>
        </div>
        <button className="btn btn-primary w-full" disabled={!ready || capturing} onClick={doCapture}>
          {capturing ? 'Capturing…' : 'Capture'}
        </button>
        {message && <div className="text-xs font-mono mt-2 text-(--color-text-dim)">{message}</div>}
        {result && (
          <div className="text-xs font-mono mt-2 p-2 rounded bg-black/30 space-y-0.5">
            <div>t = [{result.translation.map((v) => v.toFixed(3)).join(', ')}]</div>
            <div>q = [{result.rotation_quat.map((v) => v.toFixed(3)).join(', ')}]</div>
            <div className="text-(--color-amber) pt-1">Restart its TF broadcaster from Node Control to apply.</div>
          </div>
        )}
      </div>
    </div>
  )
}

export function CalibrationWizard() {
  const [generating, setGenerating] = useState(false)
  const [boardMsg, setBoardMsg] = useState<string | null>(null)
  const [boardReady, setBoardReady] = useState(false)

  async function generate() {
    setGenerating(true)
    try {
      const res = await api.generateBoard()
      setBoardMsg(res.message)
      setBoardReady(true)
    } catch (e) {
      setBoardMsg(String(e))
    } finally {
      setGenerating(false)
    }
  }

  return (
    <div className="panel">
      <div className="panel-header">
        <h2 className="font-semibold text-sm tracking-tight">Camera Calibration</h2>
        <div className="flex gap-2">
          <button className="btn" disabled={generating} onClick={generate}>
            {generating ? 'Generating…' : '📄 Generate Board'}
          </button>
          {boardReady && (
            <a className="btn btn-primary" href={boardPngUrl()} download="charuco_board.png">
              ⬇ Download
            </a>
          )}
        </div>
      </div>
      {boardMsg && (
        <pre className="px-4 py-2 text-xs font-mono text-(--color-text-dim) whitespace-pre-wrap border-b border-(--color-border)">
          {boardMsg}
        </pre>
      )}
      <div className="p-4">
        <p className="text-xs text-(--color-text-dim) mb-3">
          Wrist-anchored, one camera at a time — position the board so the wrist camera and the
          target camera both see it (jog the arm as needed). The tile border turns green the
          moment both do; Capture becomes enabled at that point.
        </p>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          {TARGETS.map((t) => (
            <CameraCapturePanel key={t.id} camera={t.id} label={t.label} />
          ))}
        </div>
      </div>
    </div>
  )
}
