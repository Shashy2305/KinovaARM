import { useEffect, useRef, useState } from 'react'
import * as THREE from 'three'
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js'

/**
 * The real fusion view: every camera's point cloud, decoded and
 * transformed into base_link on the backend (ros_bridge.get_fused_points),
 * rendered here as ONE merged 3D point cloud you can orbit/zoom --
 * unlike MultiCameraView's 3 tiles, which are just three images shown
 * next to each other, this is actual sensor fusion: positions registered
 * into one consistent frame via the same calibration the rest of the
 * pipeline uses.
 */
type FusionMode = 'rgb' | 'camera' | 'overlap'
interface FusionStats {
  cameras: Record<string, { status: string; points: number; size?: [number, number] }>
  pairs: Record<string, { median_nn_cm: number | null; overlap_pct: number; shared_points: number }>
  merged: number
  ms: number
}
const CAM_LABEL: Record<string, string> = { realsense: 'RS1', realsense2: 'RS2', realsense3: 'RS3', wrist: 'Wrist' }
const CAM_DOT: Record<string, string> = { realsense: '#ff594d', realsense2: '#4de666', realsense3: '#598cff', wrist: '#ffd94d' }
const MODES: { id: FusionMode; label: string; hint: string }[] = [
  { id: 'rgb', label: 'Colour', hint: 'the cameras\' own colours' },
  { id: 'camera', label: 'By camera', hint: 'one colour per camera, WHITE where two or more agree: a mis-calibrated camera shows up as a separate coloured layer' },
  { id: 'overlap', label: 'Overlap', hint: 'grey = one camera, yellow = two, green = three or more' },
]

export function FusionView() {
  const [mode, setMode] = useState<FusionMode>('rgb')
  const [stats, setStats] = useState<FusionStats | null>(null)
  const mountRef = useRef<HTMLDivElement>(null)

  useEffect(() => {
    let alive = true
    const load = async () => {
      try {
        const r = await fetch('/api/fusion/stats')
        if (alive && r.ok) setStats(await r.json())
      } catch { /* backend restarting */ }
    }
    load()
    const id = setInterval(load, 2000)
    return () => { alive = false; clearInterval(id) }
  }, [])

  const [pointCount, setPointCount] = useState(0)
  const [connected, setConnected] = useState(false)
  const [lastUpdate, setLastUpdate] = useState<number | null>(null)

  useEffect(() => {
    const mount = mountRef.current
    if (!mount) return

    const scene = new THREE.Scene()
    scene.background = new THREE.Color(0x09090f)

    const camera = new THREE.PerspectiveCamera(55, mount.clientWidth / mount.clientHeight, 0.01, 50)
    camera.position.set(0.8, -0.8, 0.9)
    camera.up.set(0, 0, 1) // base_link: z-up

    const renderer = new THREE.WebGLRenderer({ antialias: true })
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2))
    renderer.setSize(mount.clientWidth, mount.clientHeight)
    mount.appendChild(renderer.domElement)

    const controls = new OrbitControls(camera, renderer.domElement)
    controls.target.set(0.3, 0, 0.2)
    controls.enableDamping = true
    controls.dampingFactor = 0.08

    // base_link reference: grid on the table plane + small axes at the origin
    const grid = new THREE.GridHelper(1.6, 16, 0x8b5cf6, 0x22222c)
    grid.rotateX(Math.PI / 2) // GridHelper is XZ by default; we want XY (z-up)
    scene.add(grid)
    const axes = new THREE.AxesHelper(0.15)
    scene.add(axes)

    const geometry = new THREE.BufferGeometry()
    const material = new THREE.PointsMaterial({ size: 0.006, vertexColors: true })
    const pointsObj = new THREE.Points(geometry, material)
    scene.add(pointsObj)

    let raf = 0
    function animate() {
      raf = requestAnimationFrame(animate)
      controls.update()
      renderer.render(scene, camera)
    }
    animate()

    function onResize() {
      if (!mount) return
      camera.aspect = mount.clientWidth / mount.clientHeight
      camera.updateProjectionMatrix()
      renderer.setSize(mount.clientWidth, mount.clientHeight)
    }
    const resizeObserver = new ResizeObserver(onResize)
    resizeObserver.observe(mount)

    // ── WS: binary interleaved [x,y,z,r,g,b, ...] float32, see fusion.py ──
    let ws: WebSocket
    let closedByUs = false
    let retryTimer: ReturnType<typeof setTimeout>
    let retry = 0

    function connect() {
      const proto = window.location.protocol === 'https:' ? 'wss' : 'ws'
      ws = new WebSocket(`${proto}://${window.location.host}/api/ws/fusion?mode=${mode}`)
      ws.binaryType = 'arraybuffer'

      ws.onopen = () => { setConnected(true); retry = 0 }
      ws.onmessage = (ev) => {
        const flat = new Float32Array(ev.data as ArrayBuffer)
        const n = flat.length / 6
        const positions = new Float32Array(n * 3)
        const colors = new Float32Array(n * 3)
        for (let i = 0; i < n; i++) {
          positions[i * 3] = flat[i * 6]
          positions[i * 3 + 1] = flat[i * 6 + 1]
          positions[i * 3 + 2] = flat[i * 6 + 2]
          colors[i * 3] = flat[i * 6 + 3]
          colors[i * 3 + 1] = flat[i * 6 + 4]
          colors[i * 3 + 2] = flat[i * 6 + 5]
        }
        geometry.setAttribute('position', new THREE.BufferAttribute(positions, 3))
        geometry.setAttribute('color', new THREE.BufferAttribute(colors, 3))
        geometry.computeBoundingSphere()
        setPointCount(n)
        setLastUpdate(Date.now())
      }
      ws.onclose = () => {
        setConnected(false)
        if (closedByUs) return
        const delay = Math.min(1000 * 2 ** retry, 8000)
        retry += 1
        retryTimer = setTimeout(connect, delay)
      }
      ws.onerror = () => ws.close()
    }
    connect()

    return () => {
      closedByUs = true
      clearTimeout(retryTimer)
      ws?.close()
      cancelAnimationFrame(raf)
      resizeObserver.disconnect()
      renderer.dispose()
      geometry.dispose()
      material.dispose()
      mount.removeChild(renderer.domElement)
    }
  }, [mode])

  const pairVerdict = (cm: number | null) => (cm === null ? 'text-(--color-text-faint)' : cm <= 1.5 ? 'text-(--color-green)' : cm <= 3 ? 'text-(--color-amber)' : 'text-(--color-red)')
  const stale = !lastUpdate || Date.now() - lastUpdate > 3000

  return (
    <div className="panel overflow-hidden">
      <div className="panel-header">
        <div>
          <h2 className="font-semibold text-sm tracking-tight">3D Fusion</h2>
          <p className="text-xs text-(--color-text-dim) mt-0.5">
            Every camera merged into one stable point cloud in base_link (fixed pixel grid, 3-frame depth median, flying pixels removed, 1 cm voxels, workspace crop)
          </p>
        </div>
        <div className="flex items-center gap-2">
          {MODES.map((m) => (
            <button key={m.id} title={m.hint} onClick={() => setMode(m.id)}
              className={`px-2.5 py-1 text-[0.72rem] font-medium border transition-colors ${
                mode === m.id ? 'border-(--color-amber) text-(--color-amber)' : 'border-(--color-border-bright) text-(--color-text-dim) hover:text-(--color-text)'}`}>
              {m.label}
            </button>
          ))}
          <span className="text-xs font-mono text-(--color-text-dim)">{pointCount.toLocaleString()} pts</span>
          <span className={`live-dot ${connected && !stale ? 'text-(--color-green)' : 'text-(--color-red)'}`} style={{ background: 'currentColor' }} />
        </div>
      </div>
      <div ref={mountRef} className="w-full" style={{ height: '560px' }} />
      {stats && (
        <div className="px-4 py-2.5 border-t border-(--color-border) text-xs space-y-1.5">
          <div className="flex flex-wrap gap-x-4 gap-y-1 font-mono">
            {Object.entries(stats.cameras).map(([name, c]) => (
              <span key={name} className={c.status === 'ok' ? 'text-(--color-text-dim)' : 'text-(--color-amber)'}>
                <span style={{ color: CAM_DOT[name] }}>●</span> {CAM_LABEL[name] ?? name}: {c.status === 'ok' ? `${c.points.toLocaleString()} pts${c.size ? ` (${c.size[0]}x${c.size[1]})` : ''}` : c.status}
              </span>
            ))}
            <span className="text-(--color-text-faint)">merged {stats.merged.toLocaleString()} voxels in {stats.ms} ms</span>
          </div>
          {Object.keys(stats.pairs).length > 0 && (
            <div className="flex flex-wrap gap-x-5 gap-y-1 font-mono">
              <span className="text-(--color-text-faint)">Do the cameras agree? median distance to the other camera's surface:</span>
              {Object.entries(stats.pairs).map(([k, v]) => {
                const [a, b] = k.split('|')
                return (
                  <span key={k} className={pairVerdict(v.median_nn_cm)}>
                    {CAM_LABEL[a] ?? a} / {CAM_LABEL[b] ?? b}: {v.median_nn_cm === null ? 'no shared view' : `${v.median_nn_cm} cm, ${v.overlap_pct}% within 3 cm`}
                  </span>
                )
              })}
            </div>
          )}
        </div>
      )}
      <div className="px-4 py-2.5 text-xs text-(--color-text-faint) border-t border-(--color-border)">
        Drag to orbit, scroll to zoom. Violet grid is the table plane through base_link; axes mark
        the origin (red=x, green=y, blue=z). Green numbers (1.5 cm or less between two cameras) mean their calibrations agree; red means one of them is off, and "By camera" shows which.
      </div>
    </div>
  )
}
