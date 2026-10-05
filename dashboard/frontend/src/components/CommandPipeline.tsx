import { useEffect, useRef, useState } from 'react'
import {
  applyEvent, stageSeconds, STAGE_ORDER,
  type Run, type StageId, type StatusEvent,
} from '../lib/pipeline'

const LABEL: Record<StageId, string> = {
  listen: 'Listen', transcribe: 'Transcribe', plan: 'Plan', execute: 'Verify + Move',
}

/** The pipeline a command really goes through, rebuilt from the ordered
 * status messages the nodes publish (see lib/pipeline.ts). Timers show real
 * wall-clock time, so "Plan 12.4 s" is the actual LLM latency. */
export function CommandPipeline({ events }: { events: StatusEvent[] }) {
  const [run, setRun] = useState<Run | null>(null)
  const [now, setNow] = useState(() => performance.now())
  const lastC = useRef(0)

  useEffect(() => {
    const fresh = events.filter((e) => (e.cseq ?? 0) > lastC.current)
    if (!fresh.length) return
    lastC.current = fresh[fresh.length - 1].cseq ?? lastC.current
    const rt = performance.now()
    setRun((prev) => fresh.reduce<Run | null>((r, e) => applyEvent(r, e, rt), prev))
  }, [events])

  const active = !!run && STAGE_ORDER.some((s) => run.stages[s].state === 'active')
  useEffect(() => {
    if (!active) return
    const id = setInterval(() => setNow(performance.now()), 100)
    return () => clearInterval(id)
  }, [active])

  if (!run) {
    return (
      <div className="border border-dashed border-(--color-border-bright) px-3 py-4 text-center text-xs text-(--color-text-faint) font-mono">
        PIPELINE IDLE — type or speak a command
      </div>
    )
  }

  const failed = STAGE_ORDER.some((s) => run.stages[s].state === 'failed')
  const finished = run.stages.execute.state === 'done'

  return (
    <div className="border border-(--color-border) bg-(--color-bg)">
      <div className="flex items-center justify-between px-3 py-2 border-b border-(--color-border)">
        <span className="font-mono text-xs truncate">
          <span className="text-(--color-text-faint)">{run.voice ? 'VOICE' : 'TEXT'} › </span>
          {run.command || '…'}
        </span>
        <span className={`label shrink-0 ${failed ? '!text-(--color-red)' : finished ? '!text-(--color-green)' : '!text-(--color-amber)'}`}>
          {failed ? 'stopped' : finished ? 'complete' : 'working'}
        </span>
      </div>

      <ol className="grid grid-cols-4 gap-px bg-(--color-border)">
        {STAGE_ORDER.map((id) => {
          const st = run.stages[id]
          const tone =
            st.state === 'failed' ? 'var(--color-red)'
              : st.state === 'done' ? 'var(--color-green)'
                : st.state === 'active' ? 'var(--color-amber)' : 'var(--color-text-faint)'
          const secs = stageSeconds(st, now).toFixed(1)
          return (
            <li key={id} className="bg-(--color-bg) px-3 pt-2.5 pb-3">
              <div className="label" style={{ color: st.state === 'skip' ? undefined : tone }}>{LABEL[id]}</div>
              <div className="font-mono text-sm mt-1 h-5" style={{ color: tone }}>
                {st.state === 'skip' && <span className="text-(--color-text-faint)">—</span>}
                {st.state === 'pending' && <span className="text-(--color-text-faint)">waiting</span>}
                {st.state === 'active' && `${secs} s`}
                {st.state === 'done' && `${secs} s ✓`}
                {st.state === 'failed' && 'failed'}
              </div>
              <div className="mt-2 h-[3px]">
                {st.state === 'active' ? <div className="sweep h-full" />
                  : <div className="h-full" style={{
                    background: st.state === 'done' ? 'var(--color-green)' : st.state === 'failed' ? 'var(--color-red)' : '#262b30',
                    opacity: st.state === 'skip' ? 0.4 : 1 }} />}
              </div>
            </li>
          )
        })}
      </ol>

      {run.steps.length > 0 && (
        <div className="px-3 py-2 border-t border-(--color-border) font-mono text-xs space-y-0.5">
          {run.steps.map((s) => (
            <div key={s.index} className="text-(--color-text-dim)">
              <span className="text-(--color-text-faint)">{s.index}/{s.total}</span> {s.action}
            </div>
          ))}
        </div>
      )}
      {run.failure && (
        <div className="px-3 py-2 border-t border-(--color-red)/50 bg-(--color-red)/10 text-xs font-mono text-(--color-red) break-words">
          {run.failure}
        </div>
      )}
    </div>
  )
}
