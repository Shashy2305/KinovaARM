import { useCallback, useEffect, useRef, useState } from 'react'
import { api, type ArmCalibResult, type ArmCalibState } from '../lib/api'

const CAMERAS = [
  { id: 'oakd', label: 'OAK-D' },
  { id: 'realsense', label: 'RealSense' },
  { id: 'realsense2', label: 'RealSense 2' },
] as const
type CameraId = (typeof CAMERAS)[number]['id']

/** Calibrate a static camera by clicking known points of the ROBOT ARM. The
 * arm's pose is known exactly from its joint angles, so each click pairs a
 * pixel with a true base_link position. The result is only a CANDIDATE file. */
export function ArmCalibration() {
  const [camera, setCamera] = useState<CameraId>('realsense')
  const [state, setState] = useState<ArmCalibState | null>(null)
  const [result, setResult] = useState<ArmCalibResult | null>(null)
  const [active, setActive] = useState<string | null>(null)
  const [msg, setMsg] = useState<string | null>(null)
  const [frame, setFrame] = useState(() => Date.now())
  const [busy, setBusy] = useState(false)
  const imgRef = useRef<HTMLImageElement>(null)

  const load = useCallback(async () => {
    try {
      const s = await api.armCalibState(camera)
      setState(s)
      setActive((a) => a ?? s.landmarks.find((l) => !l.clicked)?.id ?? null)
    } catch (e) {
      setMsg(String(e))
      setState(null)
    }
  }, [camera])

  useEffect(() => { setResult(null); setActive(null); setMsg(null); setFrame(Date.now()); load() }, [camera, load])

  // The backend creates the calibration node on first use, and TF / intrinsics /
  // a camera frame take a few seconds to arrive: keep retrying until all are ready.
  const ready = !!state && state.arm_ok && state.k_ok && state.current_extrinsic_ok
  useEffect(() => {
    if (ready) return
    const id = setInterval(() => { load(); setFrame(Date.now()) }, 2500)
    return () => clearInterval(id)
  }, [ready, load])

  const clickedCount = state?.landmarks.filter((l) => l.clicked).length ?? 0
  const [W, H] = state?.image ?? [1280, 720]

  async function onImageClick(e: React.MouseEvent<HTMLImageElement>) {
    if (!active || !state) return
    const r = e.currentTarget.getBoundingClientRect()
    const u = ((e.clientX - r.left) * W) / r.width
    const v = ((e.clientY - r.top) * H) / r.height
    await api.armCalibClick(camera, active, u, v)
    const s = await api.armCalibState(camera)
    setState(s)
    setResult(null)
    setActive(s.landmarks.find((l) => !l.clicked)?.id ?? null)
  }

  async function run(fn: () => Promise<void>) {
    setBusy(true)
    try { await fn() } catch (e) { setMsg(String(e)) } finally { setBusy(false) }
  }

  const sol = result?.solved_projection

  return (
    <div className="panel">
      <div className="panel-header">
        <h2>Calibrate a camera from the arm</h2>
        <div className="flex gap-2">
          <button className="btn" disabled={busy} onClick={() => run(async () => { await api.armCalibReset(camera); setResult(null); setActive(null); await load() })}>Reset</button>
          <button className="btn" disabled={busy} onClick={() => { setFrame(Date.now()); load() }}>New frame</button>
          <button className="btn btn-primary" disabled={busy || clickedCount < 5}
            onClick={() => run(async () => setResult(await api.armCalibSolve(camera)))}>
            {`Solve (${clickedCount}/5+)`}
          </button>
        </div>
      </div>
      <div className="p-4 space-y-4">
        <div className="flex gap-2">
          {CAMERAS.map((c) => (
            <button key={c.id} onClick={() => setCamera(c.id)}
              className={`px-3 py-1 text-[0.78rem] font-medium border transition-colors ${
                camera === c.id ? 'border-(--color-amber) text-(--color-amber)' : 'border-(--color-border-bright) text-(--color-text-dim) hover:text-(--color-text)'}`}>
              {c.label}
            </button>
          ))}
        </div>

        <p className="text-xs text-(--color-text-dim) leading-relaxed">
          Do <b>not move the arm</b> while clicking. Choose a pose where the arm is <b>extended</b> and the gripper is visible
          (a straight-up arm gives collinear points and a poor result). Select a landmark on the right, click it on the image,
          and repeat for at least 5. Skip any that are hidden. <span className="text-(--color-red)">Red</span> circles are where the
          <i> current</i> calibration thinks each landmark is; if they are far from the real arm, the calibration is wrong.
        </p>

        {msg && <div className="text-xs font-mono text-(--color-red)">{msg}</div>}

        <div className="grid gap-4 xl:grid-cols-[minmax(0,1fr)_20rem]">
          <div className="relative border border-(--color-border) bg-black select-none">
            <img ref={imgRef} key={`${camera}-${frame}`} src={`/api/camera/${camera}/snapshot.jpg?t=${frame}`}
              alt={camera} className="block w-full h-auto cursor-crosshair" onClick={onImageClick} draggable={false} />
            <svg viewBox={`0 0 ${W} ${H}`} className="absolute inset-0 w-full h-full pointer-events-none">
              {state?.landmarks.map((l, i) => (
                <g key={l.id}>
                  {l.projected_now && (
                    <g stroke="#ef5b5b" fill="none" strokeWidth={Math.max(2, W / 400)}>
                      <circle cx={l.projected_now[0]} cy={l.projected_now[1]} r={W / 90} />
                      <text x={l.projected_now[0] + W / 80} y={l.projected_now[1]} fill="#ef5b5b" stroke="none" fontSize={W / 55}>{l.label.split(' ')[0]}</text>
                    </g>
                  )}
                  {sol?.[l.id] && (
                    <circle cx={sol[l.id][0]} cy={sol[l.id][1]} r={W / 110} fill="none" stroke="#5aa9e6" strokeWidth={Math.max(2, W / 400)} />
                  )}
                  {l.clicked && (
                    <g>
                      <circle cx={l.clicked[0]} cy={l.clicked[1]} r={W / 150} fill="#4cc38a" />
                      <text x={l.clicked[0] + W / 120} y={l.clicked[1] - W / 120} fill="#4cc38a" fontSize={W / 50} fontFamily="monospace">{i + 1}</text>
                    </g>
                  )}
                </g>
              ))}
            </svg>
          </div>

          <div className="space-y-3 text-xs">
            {!ready && (
              <div className="text-(--color-amber)">
                Waiting for {!state ? 'the backend' : !state.arm_ok ? 'the arm pose (is the arm driver publishing? RUNBOOK section 7)'
                  : !state.k_ok ? 'camera intrinsics' : 'the camera frame'}…
              </div>
            )}
            <ol className="space-y-1">
              {state?.landmarks.map((l, i) => (
                <li key={l.id}>
                  <button onClick={() => setActive(l.id)}
                    className={`w-full text-left border px-2.5 py-1.5 ${active === l.id ? 'border-(--color-amber)' : 'border-(--color-border)'} hover:border-(--color-border-bright)`}>
                    <div className="flex items-center justify-between">
                      <span className="font-semibold"><span className="font-mono text-(--color-text-faint) mr-1.5">{i + 1}</span>{l.label}</span>
                      <span className={l.clicked ? 'text-(--color-green)' : 'text-(--color-text-faint)'}>{l.clicked ? 'clicked' : 'to click'}</span>
                    </div>
                    {active === l.id && <div className="mt-1 text-(--color-text-dim) leading-snug">{l.hint}</div>}
                    {result?.reproj_px?.[l.id] !== undefined && (
                      <div className={`mt-1 font-mono ${result.suspect === l.id ? 'text-(--color-red)' : 'text-(--color-text-dim)'}`}>
                        error {result.reproj_px[l.id].toFixed(1)} px{result.suspect === l.id ? ' - looks like a mis-click' : ''}
                      </div>
                    )}
                  </button>
                </li>
              ))}
            </ol>

            {result?.error && <div className="text-(--color-red) font-mono">{result.error}</div>}
            {result?.ok && (
              <div className="border border-(--color-border) p-3 space-y-1.5">
                <div className="label">Result (cyan circles on the image)</div>
                <div className="font-mono">reprojection RMS {result.rms_px?.toFixed(1)} px</div>
                {result.vs_current && (
                  <div className="font-mono">
                    vs current calibration: rotated {result.vs_current.rotation_deg.toFixed(1)}°, moved {(result.vs_current.translation_m * 100).toFixed(0)} cm
                  </div>
                )}
                {result.warnings?.map((w, i) => <div key={i} className="text-(--color-amber)">{w}</div>)}
                <button className="btn btn-primary w-full mt-1" disabled={busy}
                  onClick={() => run(async () => { const r = await api.armCalibSave(camera); setMsg(r.message) })}>
                  Save as candidate (does not change the live calibration)
                </button>
              </div>
            )}
            {msg && !msg.startsWith('Error') && <div className="font-mono text-(--color-green) break-all">{msg}</div>}
          </div>
        </div>
      </div>
    </div>
  )
}
