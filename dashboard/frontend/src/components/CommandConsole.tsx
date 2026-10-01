import { useMemo, useState } from 'react'
import { api } from '../lib/api'
import type { RosStatus } from '../lib/types'
import { CommandPipeline } from './CommandPipeline'

function StatusLine({ label, value }: { label: string; value: string | null }) {
  const lower = (value ?? '').toLowerCase()
  const color = lower.includes('reject') || lower.includes('error') || lower.includes('failed')
    ? 'text-(--color-red)'
    : lower.includes('executing') || lower.includes('planning')
      ? 'text-(--color-amber)'
      : lower.includes('complete') || lower.includes('ready')
        ? 'text-(--color-green)'
        : 'text-(--color-text-dim)'
  return (
    <div className="flex gap-2 text-xs font-mono">
      <span className="text-(--color-text-faint) w-24 shrink-0">{label}</span>
      <span className={color}>{value ?? '—'}</span>
    </div>
  )
}

const BUILT_IN_SUGGESTIONS = ['go home', 'go left', 'go right']

export function CommandConsole({ ros, scene }: { ros: RosStatus | null; scene: RosStatus['scene_snapshot'] }) {
  const [text, setText] = useState('')
  const [sending, setSending] = useState(false)
  const [history, setHistory] = useState<string[]>([])

  // Dynamic suggestions from whatever's actually visible right now, not a
  // hardcoded object list -- if the detector can't see something (it has
  // no "pen" class, for instance -- COCO-80 doesn't have one), it simply
  // won't appear here rather than offering a command that would fail.
  const objectSuggestions = useMemo(() => {
    if (!scene) return []
    const labels = new Set<string>()
    for (const obj of Object.values(scene)) {
      if (obj.reachable && !obj.stale) labels.add(obj.label)
    }
    return [...labels].slice(0, 6).map((l) => `pick up the ${l}`)
  }, [scene])

  async function sendText(t: string) {
    if (!t.trim()) return
    setSending(true)
    try {
      await api.sendCommand(t)
      setHistory((h) => [t, ...h].slice(0, 10))
    } finally {
      setSending(false)
    }
  }

  return (
    <div className="panel">
      <div className="panel-header">
        <h2 className="font-semibold text-sm tracking-tight">Command Console</h2>
      </div>
      <div className="p-5 space-y-4">
        <div className="flex gap-2">
          <input
            className="flex-1 bg-black/30 border border-(--color-border-bright) rounded-xl px-3.5 py-2.5 text-sm outline-none focus:border-(--color-brand) transition-colors"
            value={text}
            onChange={(e) => setText(e.target.value)}
            onKeyDown={(e) => e.key === 'Enter' && sendText(text)}
            placeholder="Type a command, or tap a suggestion below…"
          />
          <button className="btn btn-primary" disabled={sending} onClick={() => sendText(text)}>
            {sending ? 'Sending…' : 'Send'}
          </button>
        </div>

        <div className="flex flex-wrap gap-2">
          {BUILT_IN_SUGGESTIONS.map((s) => (
            <button key={s} className="suggestion-chip" onClick={() => sendText(s)}>
              {s}
            </button>
          ))}
          {objectSuggestions.map((s) => (
            <button key={s} className="suggestion-chip" onClick={() => sendText(s)}>
              ✦ {s}
            </button>
          ))}
          {objectSuggestions.length === 0 && (
            <span className="text-xs text-(--color-text-faint) py-1.5">
              no reachable objects detected yet
            </span>
          )}
        </div>

        <CommandPipeline ros={ros} />

        <details className="text-xs">
          <summary className="text-(--color-text-faint) cursor-pointer select-none">Raw status (debug)</summary>
          <div className="space-y-1.5 rounded-xl bg-black/20 p-3.5 mt-2">
            <StatusLine label="planner" value={ros?.planner_status ?? null} />
            <StatusLine label="pick/place" value={ros?.pick_place_status ?? null} />
            <StatusLine label="arm" value={ros?.arm_status ?? null} />
          </div>
        </details>

        {history.length > 0 && (
          <div>
            <div className="text-xs text-(--color-text-dim) mb-1.5">Recent commands</div>
            <div className="space-y-1">
              {history.map((h, i) => (
                <div key={i} className="text-xs font-mono text-(--color-text-dim)">› {h}</div>
              ))}
            </div>
          </div>
        )}
      </div>
    </div>
  )
}
