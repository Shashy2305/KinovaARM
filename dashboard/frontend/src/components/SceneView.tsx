import { useEffect, useState } from 'react'
import { api } from '../lib/api'
import type { RosStatus, SceneObject, UnknownObstacle } from '../lib/types'

// Fixed view window in metres (base_link frame) -- generous enough to show
// both the reachable workspace and typical background-clutter outliers
// without the plot rescaling every frame and feeling jittery.
const VIEW_X: [number, number] = [-0.3, 1.8]
const VIEW_Y: [number, number] = [-1.0, 1.6]
const W = 480
const H = 320

function toPx(x: number, y: number): [number, number] {
  const px = ((x - VIEW_X[0]) / (VIEW_X[1] - VIEW_X[0])) * W
  const py = H - ((y - VIEW_Y[0]) / (VIEW_Y[1] - VIEW_Y[0])) * H
  return [px, py]
}

export function SceneView({ scene, unknown }: { scene: RosStatus['scene_snapshot']; unknown?: UnknownObstacle[] | null }) {
  const [bounds, setBounds] = useState<{ x: [number, number]; y: [number, number] } | null>(null)

  useEffect(() => {
    api.boundaryCurrent().then((d) => {
      if (d.exists) setBounds({ x: d.x as [number, number], y: d.y as [number, number] })
    }).catch(() => {})
  }, [])

  // Only objects seen recently: stale entries are ghosts of things that moved or were never there.
  const all = scene ? Object.entries(scene) : []
  const objects = all.filter(([, o]) => !o.stale)
  const staleCount = all.length - objects.length
  const [origin_x, origin_y] = toPx(0, 0)

  return (
    <div className="panel">
      <div className="panel-header">
        <h2 className="font-semibold text-sm tracking-tight">Scene (top-down, base_link)</h2>
        <span className="text-xs font-mono text-(--color-text-dim)">{objects.length} on the table{staleCount > 0 ? ` (+${staleCount} expiring)` : ''}</span>
      </div>
      <div className="p-4">
        <svg viewBox={`0 0 ${W} ${H}`} className="w-full rounded-lg bg-black/30 border border-(--color-border)">
          {/* grid */}
          {Array.from({ length: 10 }).map((_, i) => (
            <line key={`vx${i}`} x1={(i / 9) * W} y1={0} x2={(i / 9) * W} y2={H} stroke="#ffffff08" />
          ))}
          {Array.from({ length: 7 }).map((_, i) => (
            <line key={`hz${i}`} x1={0} y1={(i / 6) * H} x2={W} y2={(i / 6) * H} stroke="#ffffff08" />
          ))}

          {/* reachable workspace rectangle */}
          {bounds && (() => {
            const [x0, y0] = toPx(bounds.x[0], bounds.y[0])
            const [x1, y1] = toPx(bounds.x[1], bounds.y[1])
            return (
              <rect
                x={Math.min(x0, x1)} y={Math.min(y0, y1)}
                width={Math.abs(x1 - x0)} height={Math.abs(y1 - y0)}
                fill="rgba(56,189,248,0.06)" stroke="var(--color-cyan)" strokeDasharray="4 3"
              />
            )
          })()}

          {/* base_link origin */}
          <circle cx={origin_x} cy={origin_y} r={5} fill="var(--color-amber)" />
          <text x={origin_x + 8} y={origin_y - 6} fontSize="9" fill="var(--color-amber)" fontFamily="monospace">
            base_link
          </text>

          {/* things on the table the detectors cannot name (from depth): solid = seen by 2 cameras, faint = one camera */}
          {(unknown ?? []).map((o, i) => {
            const [px, py] = toPx(o.x, o.y)
            const wpx = Math.max(6, (Math.max(o.w, o.h) / (VIEW_X[1] - VIEW_X[0])) * W)
            const solid = !!o.confirmed
            return (
              <g key={`unk${i}`} opacity={solid ? 1 : 0.4}>
                <rect x={px - wpx / 2} y={py - wpx / 2} width={wpx} height={wpx} fill="rgba(251,146,60,0.25)"
                      stroke="var(--color-amber)" strokeDasharray={solid ? undefined : '3 2'} />
                <text x={px + wpx / 2 + 2} y={py + 3} fontSize="8" fill="var(--color-amber)" fontFamily="monospace">
                  ? {Math.round(o.height * 100)} cm
                </text>
              </g>
            )
          })}

          {objects.map(([id, obj]: [string, SceneObject]) => {
            const [px, py] = toPx(obj.x, obj.y)
            const color = obj.reachable ? 'var(--color-green)' : 'var(--color-text-faint)'
            return (
              <g key={id} opacity={obj.stale ? 0.35 : 1}>
                <circle cx={px} cy={py} r={obj.reachable ? 5 : 3.5} fill={color} />
                <text x={px + 6} y={py + 3} fontSize="8" fill={color} fontFamily="monospace">
                  {obj.label}
                </text>
              </g>
            )
          })}
        </svg>
        <div className="flex items-center gap-4 mt-2 text-xs font-mono text-(--color-text-dim)">
          <span><span className="inline-block w-2 h-2 rounded-full bg-(--color-green) mr-1" />reachable</span>
          <span><span className="inline-block w-2 h-2 rounded-full bg-(--color-text-faint) mr-1" />unreachable</span>
          <span><span className="inline-block w-2 h-2 rounded-full bg-(--color-amber) mr-1" />base_link</span>
          <span><span className="inline-block w-2 h-2 border border-(--color-amber) mr-1" />unknown obstacle (depth)</span>
          <span className="ml-auto border border-(--color-cyan) border-dashed px-1">workspace bound</span>
        </div>
      </div>
    </div>
  )
}
