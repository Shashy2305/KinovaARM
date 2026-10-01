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
export function FusionView() {
  const mountRef = useRef<HTMLDivElement>(null)
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
      ws = new WebSocket(`${proto}://${window.location.host}/api/ws/fusion`)
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
  }, [])

  const stale = !lastUpdate || Date.now() - lastUpdate > 3000

  return (
    <div className="panel overflow-hidden">
      <div className="panel-header">
        <div>
          <h2 className="font-semibold text-sm tracking-tight">3D Fusion</h2>
          <p className="text-xs text-(--color-text-dim) mt-0.5">
            All three cameras merged into one point cloud, registered in base_link
          </p>
        </div>
        <div className="flex items-center gap-2">
          <span className="text-xs font-mono text-(--color-text-dim)">{pointCount.toLocaleString()} pts</span>
          <span className={`live-dot ${connected && !stale ? 'text-(--color-green)' : 'text-(--color-red)'}`} style={{ background: 'currentColor' }} />
        </div>
      </div>
      <div ref={mountRef} className="w-full" style={{ height: '560px' }} />
      <div className="px-4 py-2.5 text-xs text-(--color-text-faint) border-t border-(--color-border)">
        Drag to orbit, scroll to zoom. Violet grid is the table plane through base_link; axes mark
        the origin (red=x, green=y, blue=z).
      </div>
    </div>
  )
}
