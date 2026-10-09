import { useEffect, useState } from 'react'
import { api } from '../lib/api'
import type { OutcomeSummary } from '../lib/types'

function pct(ok: number, n: number) {
  return n ? `${ok}/${n} (${Math.round((100 * ok) / n)}%)` : '-'
}

function Row({ k, v }: { k: string; v: [number, number] }) {
  return (
    <div className="flex justify-between gap-3 text-xs font-mono">
      <span className="text-(--color-text-dim)">{k}</span>
      <span>{pct(v[0], v[1])}</span>
    </div>
  )
}

// Read-only: what the robot has actually done (live runs), from ~/.ros/outcomes. Refreshes every 10 s.
export function OutcomesPanel() {
  const [data, setData] = useState<OutcomeSummary | null>(null)
  const [liveOnly, setLiveOnly] = useState(true)
  const [err, setErr] = useState('')

  useEffect(() => {
    let stop = false
    async function load() {
      try {
        const d = await api.outcomes(liveOnly)
        if (!stop) { setData(d); setErr('') }
      } catch (e) {
        if (!stop) setErr(String(e))
      }
      if (!stop) setTimeout(load, 10000)
    }
    load()
    return () => { stop = true }
  }, [liveOnly])

  const s = data?.summary
  const lessons = s ? [...s.lessons, ...(s.grasp_lessons ?? [])] : []

  return (
    <div className="panel">
      <div className="panel-header">
        <h2 className="font-semibold text-sm tracking-tight">Outcomes (what the arm has actually done)</h2>
        <label className="text-xs font-mono text-(--color-text-dim) flex items-center gap-2">
          <input type="checkbox" checked={liveOnly} onChange={(e) => setLiveOnly(e.target.checked)} />
          live runs only
        </label>
      </div>
      <div className="p-4 space-y-3">
        {err && <div className="text-xs text-(--color-amber)">outcome log unavailable: {err}</div>}
        {!data && !err && <div className="text-xs text-(--color-text-dim)">loading...</div>}
        {data && s && (
          <>
            <div className="grid grid-cols-3 gap-3 text-center">
              {([['Picks', s.picks], ['Places', s.places], ['Commands', s.commands]] as const).map(([name, v]) => (
                <div key={name} className="rounded-lg border border-(--color-border) p-2">
                  <div className="text-[10px] uppercase tracking-wider text-(--color-text-faint)">{name}</div>
                  <div className="font-mono text-sm">{pct(v.ok, v.n)}</div>
                </div>
              ))}
            </div>
            <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
              <div className="space-y-1">
                <div className="text-[10px] uppercase tracking-wider text-(--color-text-faint)">Picks by object</div>
                {Object.entries(s.by_label).sort().map(([k, v]) => <Row key={k} k={k} v={v} />)}
              </div>
              <div className="space-y-1">
                <div className="text-[10px] uppercase tracking-wider text-(--color-text-faint)">By grip strategy</div>
                {Object.entries(s.by_strategy).sort().map(([k, v]) => <Row key={k} k={k} v={v} />)}
              </div>
              <div className="space-y-1">
                <div className="text-[10px] uppercase tracking-wider text-(--color-text-faint)">By nearest neighbour</div>
                {Object.entries(s.by_neighbour).sort().map(([k, v]) => <Row key={k} k={k} v={v} />)}
              </div>
            </div>
            {Object.keys(s.grip_median).length > 0 && (
              <div className="text-xs font-mono text-(--color-text-dim)">
                typical finger reading when a grip held: {Object.entries(s.grip_median).map(([k, v]) => `${k} ${v}`).join('  ·  ')}
              </div>
            )}
            {lessons.length > 0 && (
              <div className="text-xs space-y-1">
                <div className="text-[10px] uppercase tracking-wider text-(--color-text-faint)">Lessons</div>
                {lessons.map((t, i) => <div key={i} className="text-(--color-text-dim)">- {t}</div>)}
              </div>
            )}
            {data.recent_failures.length > 0 && (
              <div className="text-xs space-y-1">
                <div className="text-[10px] uppercase tracking-wider text-(--color-text-faint)">Recent failures</div>
                {data.recent_failures.map((f, i) => (
                  <div key={i} className="font-mono text-(--color-text-dim)">
                    {f.time?.slice(5, 16)} {f.kind} {f.label ?? ''} - {f.step ?? '?'}{f.reason ? `: ${f.reason}` : ''}
                  </div>
                ))}
              </div>
            )}
            <div className="text-[10px] font-mono text-(--color-text-faint)">
              {data.n_records} records · {data.training_samples} self-labelled training samples kept
            </div>
          </>
        )}
      </div>
    </div>
  )
}
