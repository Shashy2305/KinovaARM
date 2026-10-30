import { useEffect, useRef, useState } from 'react'
import { api, type LearningMap } from '../lib/api'

/**
 * What the grasp learning believes, drawn on the table (top-down, base_link: robot at the left, x runs to the right, y runs down).
 * Colour of each cell = the LOCAL success rate of the chosen method there (red 0 .. green 1), faded where there is little nearby evidence; dots = every past pick (filled green = worked,
 * red cross = failed); blue outline = where the side grasp is allowed now (the seed zone plus the trust region that grows with successes and shrinks with failures).
 * The same numbers the arm controller prints as "learned here: ...". Read-only; refreshes every 15 s.
 */
const METHODS = ['top', 'side', 'lean'] as const
type Method = (typeof METHODS)[number]
const W = 720
const H = 300

function rateColour(p: number, evidence: number) {
  const hue = Math.max(0, Math.min(1, p)) * 120                       // 0 red .. 120 green
  const alpha = 0.12 + 0.55 * Math.min(1, evidence / 3)               // little nearby evidence -> faint
  return `hsla(${hue}, 75%, 48%, ${alpha})`
}

export function LearningPanel() {
  const [data, setData] = useState<LearningMap | null>(null)
  const [label, setLabel] = useState('cup')
  const [method, setMethod] = useState<Method>('side')
  const [err, setErr] = useState('')
  const [hover, setHover] = useState<string>('')
  const canvas = useRef<HTMLCanvasElement>(null)

  useEffect(() => {
    let stop = false
    async function load() {
      try {
        const d = await api.learning(label)
        if (!stop) { setData(d); setErr('') }
      } catch (e) {
        if (!stop) setErr(String(e))
      }
      if (!stop) setTimeout(load, 15000)
    }
    load()
    return () => { stop = true }
  }, [label])

  // geometry shared by drawing and hover
  const geo = (d: LearningMap) => {
    const x0 = d.x[0], x1 = d.x[d.x.length - 1], y0 = d.y[0], y1 = d.y[d.y.length - 1]
    const step = d.x.length > 1 ? d.x[1] - d.x[0] : 0.05
    const px = (x: number) => ((x - (x0 - step / 2)) / (x1 - x0 + step)) * W
    const py = (y: number) => ((y - (y0 - step / 2)) / (y1 - y0 + step)) * H
    return { x0, x1, y0, y1, step, px, py }
  }

  useEffect(() => {
    const c = canvas.current
    if (!c || !data) return
    const g = c.getContext('2d')!
    const { step, px, py } = geo(data)
    g.clearRect(0, 0, W, H)
    g.fillStyle = '#16191d'
    g.fillRect(0, 0, W, H)
    const cells = data.grid[method]
    const cw = px(data.x[0] + step) - px(data.x[0]), ch = py(data.y[0] + step) - py(data.y[0])
    data.x.forEach((x, i) => data.y.forEach((y, j) => {
      const [p, n] = cells[i][j]
      g.fillStyle = rateColour(p, n)
      g.fillRect(px(x - step / 2), py(y - step / 2), cw + 0.5, ch + 0.5)
    }))
    // side-grasp allowed region outline
    if (method === 'side') {
      g.strokeStyle = '#4fb4e8'
      g.lineWidth = 2
      data.x.forEach((x, i) => data.y.forEach((y, j) => {
        if (!data.side_allowed[i][j]) return
        const edges: [number, number, number, number, boolean][] = [
          [px(x - step / 2), py(y - step / 2), px(x + step / 2), py(y - step / 2), j === 0 || !data.side_allowed[i][j - 1]],
          [px(x - step / 2), py(y + step / 2), px(x + step / 2), py(y + step / 2), j === data.y.length - 1 || !data.side_allowed[i][j + 1]],
          [px(x - step / 2), py(y - step / 2), px(x - step / 2), py(y + step / 2), i === 0 || !data.side_allowed[i - 1][j]],
          [px(x + step / 2), py(y - step / 2), px(x + step / 2), py(y + step / 2), i === data.x.length - 1 || !data.side_allowed[i + 1][j]],
        ]
        for (const [ax, ay, bx, by, draw] of edges) if (draw) { g.beginPath(); g.moveTo(ax, ay); g.lineTo(bx, by); g.stroke() }
      }))
    }
    // robot base marker and axes labels
    g.fillStyle = '#f5a83a'
    g.beginPath(); g.arc(px(0.0), py(0.0), 7, 0, Math.PI * 2); g.fill()
    g.fillStyle = '#9aa3ad'; g.font = '11px monospace'
    g.fillText('robot base (x=0)', Math.max(4, px(0.0) + 8), py(0.0) - 8)
    g.fillText('x forward →', W - 90, H - 6)
    g.fillText('← left (−y)      right (+y) →', 6, H - 6)
    // grid lines every 10 cm in x
    g.strokeStyle = 'rgba(255,255,255,0.07)'; g.lineWidth = 1
    for (let x = 0.1; x <= data.x[data.x.length - 1]; x += 0.1) { g.beginPath(); g.moveTo(px(x), 0); g.lineTo(px(x), H); g.stroke() }
    // every past pick of this method
    for (const d of data.dots) {
      if (d.method !== method) continue
      const X = px(d.x), Y = py(d.y)
      if (d.ok) { g.fillStyle = '#5be39a'; g.strokeStyle = '#0b2'; g.beginPath(); g.arc(X, Y, 5, 0, Math.PI * 2); g.fill(); g.stroke() } else {
        g.strokeStyle = '#ff6b6b'; g.lineWidth = 2.5
        g.beginPath(); g.moveTo(X - 5, Y - 5); g.lineTo(X + 5, Y + 5); g.moveTo(X + 5, Y - 5); g.lineTo(X - 5, Y + 5); g.stroke()
      }
    }
  }, [data, method])

  const onMove = (e: React.MouseEvent<HTMLCanvasElement>) => {
    if (!data) return
    const r = e.currentTarget.getBoundingClientRect()
    const { x0, x1, y0, y1, step } = geo(data)
    const x = (x0 - step / 2) + ((e.clientX - r.left) / r.width) * (x1 - x0 + step)
    const y = (y0 - step / 2) + ((e.clientY - r.top) / r.height) * (y1 - y0 + step)
    const i = Math.max(0, Math.min(data.x.length - 1, Math.round((x - x0) / step)))
    const j = Math.max(0, Math.min(data.y.length - 1, Math.round((y - y0) / step)))
    const [p, n] = data.grid[method][i][j]
    setHover(`x ${data.x[i].toFixed(2)}  y ${data.y[j].toFixed(2)}:  ${method} success ${Math.round(p * 100)}%  (evidence ${n.toFixed(1)})` +
      (method === 'side' ? `  ·  side grasp ${data.side_allowed[i][j] ? 'ALLOWED' : 'not allowed'}` : ''))
  }

  const t = data?.totals[method]
  return (
    <div className="panel">
      <div className="panel-header">
        <h2 className="font-semibold text-sm tracking-tight">Learned grasp success on the table</h2>
        <div className="flex gap-3 items-center text-xs font-mono">
          <select className="btn" value={label} onChange={(e) => setLabel(e.target.value)}>
            {(data?.labels ?? [label]).map((l) => <option key={l} value={l}>{l}</option>)}
          </select>
          <div className="flex">
            {METHODS.map((m) => (
              <button key={m} className={`px-3 py-1 border ${method === m ? 'border-(--color-amber) text-(--color-amber)' : 'border-(--color-border)'}`} onClick={() => setMethod(m)}>{m}</button>
            ))}
          </div>
        </div>
      </div>
      <div className="p-4 space-y-2 text-xs">
        {err && <div className="font-mono text-(--color-red)">{err}</div>}
        <div className="font-mono text-(--color-text-dim) flex flex-wrap gap-x-6 gap-y-1">
          <span>{label} · {method}: {t ? `${t.successes}/${t.trials} picks worked` : '—'}</span>
          <span>{hover || 'move the mouse over the map for the local rate'}</span>
        </div>
        <canvas ref={canvas} width={W} height={H} onMouseMove={onMove} onMouseLeave={() => setHover('')}
          className="w-full border border-(--color-border) rounded cursor-crosshair" style={{ aspectRatio: `${W} / ${H}` }} />
        <p className="text-(--color-text-faint) leading-relaxed">
          Colour = local success rate of the chosen method (red 0% → green 100%), faint where few past picks were nearby. Dots: filled = worked, red cross = failed. For <b>side</b>, the blue outline is where the side grasp
          is allowed now: the hand-proven zone plus a trust region that grows within {data ? Math.round(data.params.trust_radius_m * 100) : 12} cm of every success and closes where the local rate falls below {data ? Math.round(data.params.block_rate * 100) : 35}%.
          The robot picks the method with the best score here; outside the outline it goes straight down.
        </p>
      </div>
    </div>
  )
}
