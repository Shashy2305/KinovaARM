import { useEffect, useRef, useState } from 'react'
import * as THREE from 'three'
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js'
import { colourHex, objectName } from '../lib/colours'

/**
 * Live 3D animation of the robot: the backend streams the position of every link (forward kinematics of the real joint states, ~15 Hz) and this
 * draws the arm and gripper moving as the real one does, over the table, the planner's workspace rectangle and the objects in the scene.
 * Read-only: nothing here can move the arm. Orbit with the mouse, wheel to zoom. Coordinates are base_link (x forward, y right, z up).
 */
interface Pose {
  ok: boolean
  reason?: string
  joints_deg?: number[]
  finger?: number | null
  links?: Record<string, number[]>
  workspace?: { x: number[]; y: number[] }
  table_top_z?: number | null
}
interface SceneObj { label: string; x: number; y: number; z: number | null; reachable?: boolean; stale?: boolean; color?: string | null }

const CHAIN = ['base_link', 'shoulder_link', 'half_arm_1_link', 'half_arm_2_link', 'forearm_link', 'spherical_wrist_1_link', 'spherical_wrist_2_link', 'bracelet_link', 'end_effector_link', 'robotiq_140_base_link']
const FINGERS: [string, string][] = [['robotiq_140_base_link', 'left_outer_finger'], ['left_outer_finger', 'left_inner_finger_pad'], ['robotiq_140_base_link', 'right_outer_finger'], ['right_outer_finger', 'right_inner_finger_pad']]
const HEIGHT: Record<string, number> = { cup: 0.09, mug: 0.09, bowl: 0.06, bottle: 0.22, mouse: 0.035, remote: 0.03, 'cell phone': 0.01, book: 0.03, scissors: 0.02, vase: 0.2 }

function segment(mesh: THREE.Mesh, a: THREE.Vector3, b: THREE.Vector3) {
  const d = new THREE.Vector3().subVectors(b, a)
  const len = d.length()
  mesh.visible = len > 1e-4
  if (!mesh.visible) return
  mesh.position.copy(a).addScaledVector(d, 0.5)
  mesh.scale.set(1, len, 1)
  mesh.quaternion.setFromUnitVectors(new THREE.Vector3(0, 1, 0), d.normalize())
}

function textSprite(text: string, color: string) {
  const c = document.createElement('canvas')
  c.width = 256; c.height = 64
  const g = c.getContext('2d')!
  g.font = 'bold 34px monospace'; g.fillStyle = color; g.textAlign = 'center'; g.textBaseline = 'middle'
  g.fillText(text, 128, 32)
  const sp = new THREE.Sprite(new THREE.SpriteMaterial({ map: new THREE.CanvasTexture(c), depthTest: false, transparent: true }))
  sp.scale.set(0.2, 0.05, 1)
  return sp
}

export function RobotView({ scene = null, height = 420 }: { scene?: Record<string, SceneObj> | null; height?: number }) {
  const mount = useRef<HTMLDivElement>(null)
  const [info, setInfo] = useState('connecting…')
  const sceneRef = useRef(scene)
  const objectsRef = useRef<THREE.Group | null>(null)
  const sceneKey = useRef('')

  useEffect(() => { sceneRef.current = scene }, [scene])

  useEffect(() => {
    const el = mount.current
    if (!el) return
    const w = el.clientWidth || 800
    const renderer = new THREE.WebGLRenderer({ antialias: true })
    renderer.setSize(w, height)
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2))
    el.appendChild(renderer.domElement)
    const three = new THREE.Scene()
    three.background = new THREE.Color(0x0e1013)
    const cam = new THREE.PerspectiveCamera(45, w / height, 0.02, 20)
    cam.up.set(0, 0, 1)
    cam.position.set(-0.9, -1.3, 1.0)
    const controls = new OrbitControls(cam, renderer.domElement)
    controls.target.set(0.35, 0, 0.25)
    controls.update()
    three.add(new THREE.AmbientLight(0xffffff, 0.75))
    const sun = new THREE.DirectionalLight(0xffffff, 1.0)
    sun.position.set(0.5, -0.8, 2)
    three.add(sun)

    // table plane + planner workspace rectangle
    const grid = new THREE.GridHelper(2, 20, 0x2a3138, 0x1a1f24)
    grid.rotation.x = Math.PI / 2
    grid.position.set(0.35, 0, -0.0125)
    three.add(grid)
    const table = new THREE.Mesh(new THREE.PlaneGeometry(1.0, 1.8), new THREE.MeshBasicMaterial({ color: 0x3b2a1f, transparent: true, opacity: 0.55 }))
    table.position.set(0.45, 0, -0.0135)
    three.add(table)
    let wsLine: THREE.LineLoop | null = null
    const setWorkspace = (wsr: { x: number[]; y: number[] }, z: number) => {
      if (wsLine) { three.remove(wsLine); wsLine.geometry.dispose() }
      const [x0, x1] = wsr.x, [y0, y1] = wsr.y
      wsLine = new THREE.LineLoop(
        new THREE.BufferGeometry().setFromPoints([[x0, y0], [x1, y0], [x1, y1], [x0, y1]].map(([x, y]) => new THREE.Vector3(x, y, z + 0.002))),
        new THREE.LineBasicMaterial({ color: 0x4fb4e8 }))
      three.add(wsLine)
    }
    three.add(new THREE.AxesHelper(0.2))
    const baseLabel = textSprite('base_link', '#f5a83a')
    baseLabel.position.set(0, 0, -0.06)
    three.add(baseLabel)

    // arm: a capsule-ish cylinder per link, a ball per joint, gripper fingers
    const armMat = new THREE.MeshStandardMaterial({ color: 0xe8edf2, metalness: 0.1, roughness: 0.5 })
    const jointMat = new THREE.MeshStandardMaterial({ color: 0x3a6ea5, metalness: 0.2, roughness: 0.4 })
    const gripMat = new THREE.MeshStandardMaterial({ color: 0x2a2d31, metalness: 0.3, roughness: 0.5 })
    const cylGeo = new THREE.CylinderGeometry(0.045, 0.045, 1, 16)
    const thinGeo = new THREE.CylinderGeometry(0.014, 0.014, 1, 10)
    const links = CHAIN.slice(1).map(() => new THREE.Mesh(cylGeo, armMat))
    const balls = CHAIN.map(() => new THREE.Mesh(new THREE.SphereGeometry(0.055, 16, 12), jointMat))
    const fingers = FINGERS.map(() => new THREE.Mesh(thinGeo, gripMat))
    links.forEach((m) => three.add(m)); balls.forEach((m) => three.add(m)); fingers.forEach((m) => three.add(m))
    const padMat = new THREE.MeshStandardMaterial({ color: 0xf5a83a })
    const pads = [0, 1].map(() => new THREE.Mesh(new THREE.BoxGeometry(0.02, 0.02, 0.06), padMat))
    pads.forEach((m) => three.add(m))
    const objects = new THREE.Group()
    three.add(objects)
    objectsRef.current = objects

    const refreshObjects = () => {
      const sc = sceneRef.current
      const key = sc ? JSON.stringify(Object.values(sc).map((o) => [o.label, o.color, Math.round(o.x * 100), Math.round(o.y * 100), o.reachable, o.stale])) : ''
      if (key === sceneKey.current) return
      sceneKey.current = key
      while (objects.children.length) { const c = objects.children.pop()!; (c as THREE.Mesh).geometry?.dispose?.() }
      if (!sc) return
      for (const o of Object.values(sc)) {
        if (o.stale) continue
        const h = HEIGHT[o.label] ?? 0.06
        const hex = colourHex(o.color)
        const col = hex ? new THREE.Color(hex) : new THREE.Color(o.reachable ? 0x4fd18b : 0x7a828c)             // its own colour when the cameras know it
        const m = new THREE.Mesh(new THREE.CylinderGeometry(0.035, 0.035, h, 20), new THREE.MeshStandardMaterial({ color: col, transparent: true, opacity: o.reachable ? 0.95 : 0.5 }))
        m.rotation.x = Math.PI / 2
        m.position.set(o.x, o.y, -0.0125 + h / 2)
        objects.add(m)
        const lab = textSprite(objectName(o.label, o.color), o.reachable ? '#4fd18b' : '#9aa3ad')
        lab.position.set(o.x, o.y, -0.0125 + h + 0.05)
        objects.add(lab)
      }
    }

    const v = (p: number[]) => new THREE.Vector3(p[0], p[1], p[2])
    let latest: Pose | null = null
    let alive = true
    let sock: WebSocket | null = null
    let retry: ReturnType<typeof setTimeout> | undefined
    const connect = () => {
      const proto = window.location.protocol === 'https:' ? 'wss' : 'ws'
      sock = new WebSocket(`${proto}://${window.location.host}/api/ws/robot`)
      sock.onmessage = (e) => { try { latest = JSON.parse(e.data) } catch { /* ignore */ } }
      sock.onclose = () => { if (alive) retry = setTimeout(connect, 1500) }
    }
    connect()

    let lastWs = ''
    let raf = 0
    let lastInfo = 0
    const loop = () => {
      raf = requestAnimationFrame(loop)
      const p = latest
      if (p?.ok && p.links) {
        const L = p.links
        CHAIN.forEach((n, i) => { if (L[n]) balls[i].position.copy(v(L[n])) })
        CHAIN.slice(1).forEach((n, i) => { if (L[n] && L[CHAIN[i]]) segment(links[i], v(L[CHAIN[i]]), v(L[n])) })
        FINGERS.forEach(([a, b], i) => { if (L[a] && L[b]) segment(fingers[i], v(L[a]), v(L[b])) })
        pads[0].position.copy(v(L.left_inner_finger_pad)); pads[1].position.copy(v(L.right_inner_finger_pad))
        if (p.workspace) {
          const key = JSON.stringify(p.workspace) + p.table_top_z
          if (key !== lastWs) { lastWs = key; setWorkspace(p.workspace, p.table_top_z ?? -0.0125) }
        }
        const now = performance.now()
        if (now - lastInfo > 250) {
          lastInfo = now
          setInfo(`live · joints ${p.joints_deg?.map((d) => d.toFixed(0)).join(' ')}° · finger ${p.finger ?? '—'}`)
        }
      } else if (p && !p.ok) setInfo(p.reason ?? 'no pose')
      refreshObjects()
      controls.update()
      renderer.render(three, cam)
    }
    loop()

    const onResize = () => {
      const nw = el.clientWidth || w
      renderer.setSize(nw, height)
      cam.aspect = nw / height
      cam.updateProjectionMatrix()
    }
    window.addEventListener('resize', onResize)
    return () => {
      alive = false
      clearTimeout(retry)
      sock?.close()
      cancelAnimationFrame(raf)
      window.removeEventListener('resize', onResize)
      controls.dispose()
      renderer.dispose()
      el.removeChild(renderer.domElement)
    }
  }, [height])

  return (
    <div className="panel overflow-hidden">
      <div className="panel-header">
        <span>Live robot (3D)</span>
        <span className="text-[11px] font-mono text-(--color-text-faint)">{info}</span>
      </div>
      <div ref={mount} style={{ height }} className="w-full" />
      <div className="px-3 py-1 text-[11px] text-(--color-text-faint)">
        Moves as the real arm moves (read-only). Drag to orbit, wheel to zoom. Blue rectangle = the planner&apos;s workspace; green = reachable objects, grey = out of range.
      </div>
    </div>
  )
}
