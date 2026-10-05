import { useEffect, useState } from 'react'
import { api, type TableStatus } from '../lib/api'

const fmt = (v: number) => v.toFixed(3)

export function TableRecorder() {
  const [st, setSt] = useState<TableStatus | null>(null)
  const [msg, setMsg] = useState<string | null>(null)
  const [busy, setBusy] = useState(false)

  useEffect(() => {
    let alive = true
    const tick = () => api.tableStatus().then((s) => alive && setSt(s)).catch(() => {})
    tick()
    const id = setInterval(tick, 1000)
    return () => { alive = false; clearInterval(id) }
  }, [])

  async function run(fn: () => Promise<{ ok?: boolean; message?: string }>) {
    setBusy(true)
    try {
      const r = await fn()
      if (r.message) setMsg(r.message)
      setSt(await api.tableStatus())
    } catch (e) {
      setMsg(String(e))
    } finally {
      setBusy(false)
    }
  }

  const n = st?.points.length ?? 0

  return (
    <div className="panel">
      <div className="panel-header">
        <h2 className="font-semibold text-sm tracking-tight">Table Safety — record the table height</h2>
        <div className="flex gap-2">
          <button className="btn" disabled={busy} onClick={() => run(async () => { await api.tableReset(); setMsg('reset'); return {} })}>
            Reset
          </button>
          <button className="btn" disabled={busy || !st?.pose} onClick={() => run(api.tableRecord)}>
            Record point
          </button>
          <button className="btn btn-primary" disabled={busy || n < 3} onClick={() => run(api.tableSave)}>
            {`Save (${n} pts)`}
          </button>
        </div>
      </div>
      <div className="p-5 space-y-4 text-sm">
        <ol className="list-decimal pl-5 text-xs text-(--color-text-dim) space-y-1">
          <li>Keep the gripper <b>open</b> and free the arm for hand-guiding (see RUNBOOK section 5). The live pose below must keep updating while you move it. Do not force an arm that is holding position.</li>
          <li>Hand-guide the arm so the very <b>tips of the open fingers touch the table surface</b>.</li>
          <li>Press <b>Record point</b>. Do this at 4 corners of the usable area plus the middle (at least 3).</li>
          <li>Press <b>Save</b>. The arm controller blocks all live motion until this is saved.</li>
        </ol>

        <div className="rounded border border-(--color-border) p-3 font-mono text-xs">
          {st?.pose
            ? `Live fingertip pose: x=${fmt(st.pose.x)}  y=${fmt(st.pose.y)}  pad z=${fmt(st.pose.z)}`
            : <span className="text-(--color-red)">{st?.pose_message ?? 'connecting…'}</span>}
        </div>

        {n > 0 && (
          <div className="font-mono text-xs text-(--color-text-dim)">
            {st!.points.map((p, i) => (
              <div key={i}>{`${i + 1}: x=${fmt(p[0])} y=${fmt(p[1])} pad z=${fmt(p[2])}`}</div>
            ))}
          </div>
        )}

        {msg && <div className="text-xs font-mono text-(--color-text-dim)">{msg}</div>}

        <div className="rounded border border-(--color-border) p-3 text-xs">
          <div className="font-semibold mb-1">Saved table geometry</div>
          {st?.saved ? (
            <div className="font-mono text-(--color-text-dim)">
              table top z={fmt(st.saved.table_top_z)} · footprint x=[{st.saved.x.join(', ')}] y=[{st.saved.y.join(', ')}]
              · rear wall x={fmt(st.saved.rear_wall_x)} · lowest allowed wrist-flange z={fmt(st.flange_floor_z)}
            </div>
          ) : (
            <div className="text-(--color-red)">
              None — live motion is blocked. ({st?.saved_message ?? '…'}) Planner falls back to a flange floor of z={st ? fmt(st.flange_floor_z) : '…'}.
            </div>
          )}
        </div>
      </div>
    </div>
  )
}
