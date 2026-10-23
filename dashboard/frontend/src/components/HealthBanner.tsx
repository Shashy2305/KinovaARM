import { useEffect, useRef, useState } from 'react'

interface FixAction { id: string; label: string; confirm: string; command: string }
interface Issue { id: string; severity: 'error' | 'warn'; title: string; detail: string; hint: string; actions?: FixAction[]; command?: string }
interface FixResult { ok: boolean; message: string }

const POLL_MS = 2500
const TIMEOUT_MS = 6000

/** What is wrong right now (hung or frozen arm feedback, driver fault, missing TF, dead cameras, an overloaded backend) in plain words with what to do about it.
 * The backend computes it (health_issues.py); a slow or failed answer from that call is itself shown as "the backend is not responding". */
export function HealthBanner() {
  const [issues, setIssues] = useState<Issue[]>([])
  const [down, setDown] = useState<{ since: number } | null>(null)
  const [open, setOpen] = useState(true)
  const misses = useRef(0)
  const [running, setRunning] = useState<string | null>(null)
  const [results, setResults] = useState<Record<string, FixResult>>({})

  useEffect(() => {
    let alive = true
    const poll = async () => {
      const ctl = new AbortController()
      const timer = setTimeout(() => ctl.abort(), TIMEOUT_MS)
      try {
        const r = await fetch('/api/health/issues', { signal: ctl.signal })
        if (!r.ok) throw new Error(String(r.status))
        const d = await r.json()
        if (alive) { setIssues(d.issues ?? []); setDown(null); misses.current = 0 }
      } catch {
        misses.current += 1
        if (alive && misses.current >= 2) setDown((d) => d ?? { since: Date.now() })
      } finally {
        clearTimeout(timer)
      }
    }
    poll()
    const id = setInterval(poll, POLL_MS)
    return () => { alive = false; clearInterval(id) }
  }, [])

  const runFix = async (issueId: string, a: FixAction) => {
    if (!window.confirm(a.confirm)) return
    setRunning(a.id)
    setResults((r) => ({ ...r, [issueId]: { ok: true, message: 'Working…' } }))
    try {
      const res = await fetch('/api/health/fix', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ action: a.id }) })
      const d: FixResult = await res.json()
      setResults((r) => ({ ...r, [issueId]: d }))
    } catch {
      setResults((r) => ({ ...r, [issueId]: { ok: a.id === 'restart_backend', message: a.id === 'restart_backend' ? 'The backend is restarting: this page reconnects in about 10 s.' : 'The request failed: the backend did not answer.' } }))
    } finally {
      setRunning(null)
    }
  }

  if (down) {
    return (
      <div className="px-5 py-2 border-b border-(--color-red)/50 bg-(--color-red)/10 text-xs">
        <span className="text-(--color-red) font-semibold">The dashboard backend is not responding. </span>
        <span className="text-(--color-text-dim)">Nothing on this page is current. It is hung, restarting or overloaded. If it does not come back within a minute, in a terminal: <span className="mono">bash scripts/restart_dashboard_backend.sh</span></span>
      </div>
    )
  }
  if (issues.length === 0) return null
  const errors = issues.filter((i) => i.severity === 'error').length
  return (
    <div className={`border-b text-xs ${errors ? 'border-(--color-red)/50 bg-(--color-red)/10' : 'border-(--color-amber)/40 bg-(--color-amber)/10'}`}>
      <button className="w-full text-left px-5 py-2 flex items-center gap-2" onClick={() => setOpen((o) => !o)}>
        <span className={`font-semibold ${errors ? 'text-(--color-red)' : 'text-(--color-amber)'}`}>
          {errors ? `${errors} problem${errors > 1 ? 's' : ''} with the robot system` : 'Warning'}
          {issues.length > errors ? ` + ${issues.length - errors} warning${issues.length - errors > 1 ? 's' : ''}` : ''}
        </span>
        <span className="text-(--color-text-faint)">{open ? '▾' : '▸'}</span>
        {!open && <span className="text-(--color-text-dim) truncate">{issues.map((i) => i.title).join(' · ')}</span>}
      </button>
      {open && (
        <ul className="px-5 pb-2 space-y-2">
          {issues.map((i) => (
            <li key={i.id} className="border-l-2 pl-3" style={{ borderColor: i.severity === 'error' ? 'var(--color-red)' : 'var(--color-amber)' }}>
              <div className={`font-semibold ${i.severity === 'error' ? 'text-(--color-red)' : 'text-(--color-amber)'}`}>{i.title}</div>
              <div className="text-(--color-text-dim) leading-snug">{i.detail}</div>
              <div className="text-(--color-text) leading-snug"><span className="text-(--color-text-faint)">What to do: </span>{i.hint}</div>
              {(i.actions?.length ?? 0) > 0 && (
                <div className="mt-1 flex flex-wrap items-center gap-2">
                  {i.actions!.map((a) => (
                    <button key={a.id} className="btn btn-primary" disabled={running !== null} onClick={() => runFix(i.id, a)}>{running === a.id ? 'Working…' : a.label}</button>
                  ))}
                  <span className="text-(--color-text-faint) mono break-all">or in a terminal: {i.actions![0].command}</span>
                </div>
              )}
              {i.command && <div className="mt-1 text-(--color-text-faint)">Check in a terminal: <span className="mono break-all">{i.command}</span></div>}
              {results[i.id] && <div className={`mt-1 font-mono ${results[i.id].ok ? 'text-(--color-green)' : 'text-(--color-red)'}`}>{results[i.id].message}</div>}
            </li>
          ))}
        </ul>
      )}
    </div>
  )
}
