import { useCallback, useEffect, useRef, useState } from 'react'
import { api, type JogState } from '../lib/api'

/** Move the arm from the dashboard: hold a button, drag the stick, or use a gamepad plugged into THIS computer (hold LB as the dead-man).
 * Every limit (speed caps, joint limits, table, rear wall) is enforced by the backend; this panel only says which way and what fraction of the cap.
 * Nothing is sent while nothing is held, and the arm stops by itself ~0.2 s after the last message. */

type Part = { joint?: { j: number; dir: number }; v?: number[] }
const TICK_MS = 80
const DEAD = 0.18
const CART_BUTTONS: { label: string; v: number[] }[] = [
  { label: 'X forward', v: [1, 0, 0, 0, 0, 0] }, { label: 'X back', v: [-1, 0, 0, 0, 0, 0] },
  { label: 'Y right', v: [0, 1, 0, 0, 0, 0] }, { label: 'Y left', v: [0, -1, 0, 0, 0, 0] },
  { label: 'Z up', v: [0, 0, 1, 0, 0, 0] }, { label: 'Z down', v: [0, 0, -1, 0, 0, 0] },
  { label: 'Roll +', v: [0, 0, 0, 1, 0, 0] }, { label: 'Roll −', v: [0, 0, 0, -1, 0, 0] },
  { label: 'Pitch +', v: [0, 0, 0, 0, 1, 0] }, { label: 'Pitch −', v: [0, 0, 0, 0, -1, 0] },
  { label: 'Yaw +', v: [0, 0, 0, 0, 0, 1] }, { label: 'Yaw −', v: [0, 0, 0, 0, 0, -1] },
]

function dz(x: number) {
  return Math.abs(x) < DEAD ? 0 : (x - Math.sign(x) * DEAD) / (1 - DEAD)
}

export function JogPanel({ compact = false }: { compact?: boolean }) {
  const [st, setSt] = useState<JogState | null>(null)
  const [phrase, setPhrase] = useState('')
  const [err, setErr] = useState<string | null>(null)
  const [mode, setMode] = useState<'cart' | 'joint'>('cart')
  const [frame, setFrame] = useState<'base' | 'tool'>('base')
  const [speed, setSpeed] = useState(0.25)
  const [pad, setPad] = useState<string | null>(null)
  const [padJoint, setPadJoint] = useState(1)
  const [wsOpen, setWsOpen] = useState(false)
  const [holding, setHolding] = useState<string | null>(null)
  const held = useRef(new Map<string, Part>())
  const ws = useRef<WebSocket | null>(null)
  const live = useRef({ mode, frame, speed, padJoint, enabled: false })
  const dpad = useRef({ l: false, r: false })
  const stick = useRef<HTMLDivElement>(null)
  const [stickPos, setStickPos] = useState<[number, number]>([0, 0])

  const isEnabled = !!st?.enabled
  useEffect(() => { live.current = { mode, frame, speed, padJoint, enabled: isEnabled } }, [mode, frame, speed, padJoint, isEnabled])

  const refresh = useCallback(async () => {
    try { setSt(await api.jogState()) } catch { /* backend restarting */ }
  }, [])
  useEffect(() => { refresh(); const id = setInterval(refresh, 1000); return () => clearInterval(id) }, [refresh])

  // websocket: opened while the jog is enabled
  useEffect(() => {
    if (!st?.enabled) { ws.current?.close(); ws.current = null; setWsOpen(false); return }
    if (ws.current) return
    const proto = window.location.protocol === 'https:' ? 'wss' : 'ws'
    const sock = new WebSocket(`${proto}://${window.location.host}/api/ws/jog`)
    sock.onopen = () => setWsOpen(true)
    sock.onclose = () => { setWsOpen(false); if (ws.current === sock) ws.current = null }
    sock.onmessage = (e) => { try { setSt(JSON.parse(e.data)) } catch { /* ignore */ } }
    ws.current = sock
  }, [st?.enabled])
  useEffect(() => () => { ws.current?.close() }, [])

  const releaseAll = useCallback(() => { held.current.clear(); setHolding(null); setStickPos([0, 0]) }, [])
  useEffect(() => {
    window.addEventListener('blur', releaseAll)
    document.addEventListener('visibilitychange', releaseAll)
    return () => { window.removeEventListener('blur', releaseAll); document.removeEventListener('visibilitychange', releaseAll) }
  }, [releaseAll])

  // the send loop: one message per tick, only while something is held
  useEffect(() => {
    const id = setInterval(() => {
      const L = live.current
      const sock = ws.current
      // gamepad
      let gv: number[] | null = null
      let gj: Part | null = null
      const gp = Array.from(navigator.getGamepads?.() ?? []).find((g) => g && g.connected) ?? null
      const padId = gp ? gp.id.slice(0, 40) : null
      setPad((p) => (p === padId ? p : padId))
      if (gp && (gp.buttons[4]?.pressed || gp.buttons[5]?.pressed)) {
        const a = gp.axes
        if (L.mode === 'cart') {
          const trig = (gp.buttons[7]?.value ?? 0) - (gp.buttons[6]?.value ?? 0)
          const roll = (gp.buttons[1]?.pressed ? 1 : 0) - (gp.buttons[2]?.pressed ? 1 : 0)
          gv = [dz(-(a[1] ?? 0)), dz(a[0] ?? 0), trig, roll, dz(-(a[3] ?? 0)), dz(a[2] ?? 0)]
        } else {
          const d = dz(-(a[1] ?? 0))
          if (d) gj = { joint: { j: L.padJoint, dir: Math.sign(d) } }
        }
      }
      if (gp) {
        const r = !!gp.buttons[15]?.pressed, l = !!gp.buttons[14]?.pressed
        if (r && !dpad.current.r) setPadJoint((j) => Math.min(7, j + 1))
        if (l && !dpad.current.l) setPadJoint((j) => Math.max(1, j - 1))
        dpad.current = { l, r }
      }
      const parts: Part[] = [...held.current.values()]
      if (gj) parts.push(gj)
      let cmd: object | null = null
      if (L.mode === 'joint') {
        const p = parts.find((x) => x.joint)
        if (p?.joint) cmd = { mode: 'joint', joint: p.joint.j, dir: p.joint.dir, speed: L.speed }
      } else {
        const sum = [0, 0, 0, 0, 0, 0]
        for (const p of parts) p.v?.forEach((x, i) => { sum[i] += x })
        if (gv) gv.forEach((x, i) => { sum[i] += x })
        if (sum.some((x) => Math.abs(x) > 1e-6)) cmd = { mode: 'cart', v: sum.map((x) => Math.max(-1, Math.min(1, x))), speed: L.speed, frame: L.frame }
      }
      if (cmd && sock && sock.readyState === WebSocket.OPEN && L.enabled) sock.send(JSON.stringify(cmd))
    }, TICK_MS)
    return () => clearInterval(id)
  }, [])

  const down = (key: string, part: Part) => (e: React.PointerEvent) => {
    e.currentTarget.setPointerCapture(e.pointerId)
    held.current.set(key, part)
    setHolding(key)
  }
  const up = (key: string) => () => { held.current.delete(key); setHolding((h) => (h === key ? null : h)) }

  const stickMove = (e: React.PointerEvent) => {
    const r = stick.current!.getBoundingClientRect()
    const nx = Math.max(-1, Math.min(1, ((e.clientX - r.left) / r.width) * 2 - 1))
    const ny = Math.max(-1, Math.min(1, ((e.clientY - r.top) / r.height) * 2 - 1))
    const x = dz(nx), y = dz(ny)
    setStickPos([nx, ny])
    held.current.set('stick', { v: [-y, x, 0, 0, 0, 0] })
  }
  const stickEnd = () => { held.current.delete('stick'); setStickPos([0, 0]) }

  const enable = async () => {
    setErr(null)
    const r = await api.jogEnable(phrase)
    if (!r.ok) setErr(r.error ?? 'refused')
    else setPhrase('')
    refresh()
  }
  const stop = async () => { releaseAll(); await api.jogStop(); refresh() }

  const enabled = !!st?.enabled
  const btn = 'btn select-none touch-none'
  return (
    <div className="panel">
      <div className="panel-header">
        <span>Jog the arm</span>
        <div className="flex gap-2 items-center">
          <span className={`text-[11px] font-mono ${enabled ? 'text-(--color-green)' : 'text-(--color-text-faint)'}`}>{enabled ? (wsOpen ? 'JOG ON' : 'connecting…') : 'JOG OFF'}</span>
          <button className="btn bg-(--color-red)/20 border-(--color-red) text-(--color-red) font-semibold" onClick={stop}>STOP</button>
        </div>
      </div>
      <div className="p-4 space-y-3 text-xs">
        {!compact && (
          <p className="text-(--color-text-dim) leading-relaxed">
            Moves the arm through the same ROS session as everything else, so it does <b>not</b> fight the driver the way the Kinova Xbox pad does (that pad uses Kinova&apos;s own session: with
            robot bringup running it can freeze the joint feedback). Do not use the Xbox pad on the arm while robot bringup is running. A gamepad plugged into <b>this computer</b> works here: hold <b>LB</b> as the
            dead-man switch. The hardware E-stop stays the real stop; this one holds position and switches the jog off.
          </p>
        )}
        {!enabled && (
          <div className="flex flex-wrap gap-2 items-center">
            <input className="btn font-mono w-72" placeholder={`type: ${st?.phrase ?? 'area clear, hands off'}`} value={phrase}
              onChange={(e) => setPhrase(e.target.value)} onKeyDown={(e) => e.key === 'Enter' && enable()} />
            <button className="btn btn-primary" onClick={enable}>Enable jog</button>
            <span className="text-(--color-text-faint)">Needs live joint states, no driver fault, an idle arm controller and the table recorded.</span>
          </div>
        )}
        {err && <div className="font-mono text-(--color-red)">{err}</div>}
        {st && (
          <div className="font-mono text-(--color-text-dim) flex flex-wrap gap-x-4 gap-y-1">
            <span>{st.message}</span>
            {st.blocked && <span className="text-(--color-red)">blocked: {st.blocked}</span>}
            {st.fault && <span className="text-(--color-red)">DRIVER FAULT</span>}
            {st.flange && <span>flange x {st.flange[0].toFixed(3)} y {st.flange[1].toFixed(3)} z {st.flange[2].toFixed(3)} m</span>}
            {st.lowest_link_above_limit_cm !== undefined && <span>lowest link {st.lowest_link_above_limit_cm} cm above its table limit</span>}
            <span>pad: {pad ?? 'none (press a button on it)'}</span>
          </div>
        )}

        <div className="flex flex-wrap gap-3 items-center">
          <div className="flex">
            {(['cart', 'joint'] as const).map((m) => (
              <button key={m} className={`px-3 py-1 border ${mode === m ? 'border-(--color-amber) text-(--color-amber)' : 'border-(--color-border)'}`} onClick={() => { releaseAll(); setMode(m) }}>
                {m === 'cart' ? 'Cartesian (x y z, roll pitch yaw)' : 'Per joint'}
              </button>
            ))}
          </div>
          {mode === 'cart' && (
            <select className="btn" value={frame} onChange={(e) => setFrame(e.target.value as 'base' | 'tool')}>
              <option value="base">axes of the robot base</option>
              <option value="tool">axes of the gripper</option>
            </select>
          )}
          <label className="flex items-center gap-2">speed {Math.round(speed * 100)}%
            <input type="range" min={0.05} max={1} step={0.05} value={speed} onChange={(e) => setSpeed(Number(e.target.value))} />
          </label>
          {st && <span className="text-(--color-text-faint)">100% = {mode === 'joint' ? `${st.speed_caps.joint_deg_s}°/s per joint` : `${st.speed_caps.linear_cm_s} cm/s, ${st.speed_caps.angular_deg_s}°/s`}</span>}
        </div>

        {mode === 'cart' ? (
          <div className="flex flex-wrap gap-6 items-start">
            <div>
              <div className="text-(--color-text-faint) mb-1">Drag: X (up/down) and Y (left/right)</div>
              <div ref={stick} className="relative w-40 h-40 rounded-full border border-(--color-border-bright) bg-white/[0.03] touch-none select-none"
                onPointerDown={(e) => { e.currentTarget.setPointerCapture(e.pointerId); stickMove(e) }}
                onPointerMove={(e) => { if (e.buttons) stickMove(e) }} onPointerUp={stickEnd} onPointerCancel={stickEnd} onLostPointerCapture={stickEnd}>
                <div className="absolute w-8 h-8 rounded-full bg-(--color-amber)/70 pointer-events-none"
                  style={{ left: `calc(50% + ${stickPos[0] * 60}px - 16px)`, top: `calc(50% + ${stickPos[1] * 60}px - 16px)` }} />
              </div>
            </div>
            <div className="grid grid-cols-2 gap-1">
              {CART_BUTTONS.map((b) => (
                <button key={b.label} className={`${btn} ${holding === b.label ? 'border-(--color-amber) text-(--color-amber)' : ''}`}
                  onPointerDown={down(b.label, { v: b.v })} onPointerUp={up(b.label)} onPointerCancel={up(b.label)} onLostPointerCapture={up(b.label)}>{b.label}</button>
              ))}
            </div>
            <div className="text-(--color-text-faint) leading-relaxed">
              Gamepad (hold LB): left stick X/Y · triggers Z · right stick yaw/pitch · X/B roll.<br />
              X is forward into the table, Y to the robot&apos;s right, Z up.
            </div>
          </div>
        ) : (
          <div className="space-y-1">
            {[1, 2, 3, 4, 5, 6, 7].map((j) => (
              <div key={j} className={`flex items-center gap-2 ${padJoint === j ? 'text-(--color-amber)' : ''}`}>
                <button className="w-16 text-left font-mono" onClick={() => setPadJoint(j)}>joint {j}</button>
                <button className={`${btn} w-14 ${holding === `j${j}-` ? 'border-(--color-amber)' : ''}`} onPointerDown={down(`j${j}-`, { joint: { j, dir: -1 } })}
                  onPointerUp={up(`j${j}-`)} onPointerCancel={up(`j${j}-`)} onLostPointerCapture={up(`j${j}-`)}>−</button>
                <button className={`${btn} w-14 ${holding === `j${j}+` ? 'border-(--color-amber)' : ''}`} onPointerDown={down(`j${j}+`, { joint: { j, dir: 1 } })}
                  onPointerUp={up(`j${j}+`)} onPointerCancel={up(`j${j}+`)} onLostPointerCapture={up(`j${j}+`)}>+</button>
                <span className="font-mono text-(--color-text-dim)">{st?.joints_deg ? `${st.joints_deg[j - 1].toFixed(1)}°` : '—'}</span>
              </div>
            ))}
            <div className="text-(--color-text-faint)">Gamepad (hold LB): d-pad left/right picks the joint (highlighted), left stick up/down moves it.</div>
          </div>
        )}
      </div>
    </div>
  )
}
