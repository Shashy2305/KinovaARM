import { useState } from 'react'
import { api } from '../lib/api'
import type { RosStatus } from '../lib/types'

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

export function CommandConsole({ ros }: { ros: RosStatus | null }) {
  const [text, setText] = useState('pick up the cup')
  const [sending, setSending] = useState(false)
  const [history, setHistory] = useState<string[]>([])

  async function send() {
    if (!text.trim()) return
    setSending(true)
    try {
      await api.sendCommand(text)
      setHistory((h) => [text, ...h].slice(0, 10))
    } finally {
      setSending(false)
    }
  }

  return (
    <div className="panel">
      <div className="panel-header">
        <h2 className="font-semibold text-sm tracking-tight">Command Console</h2>
      </div>
      <div className="p-4 space-y-3">
        <div className="flex gap-2">
          <input
            className="flex-1 bg-black/30 border border-(--color-border-bright) rounded-md px-3 py-2 font-mono text-sm outline-none focus:border-(--color-cyan)"
            value={text}
            onChange={(e) => setText(e.target.value)}
            onKeyDown={(e) => e.key === 'Enter' && send()}
            placeholder='"pick up the cup"'
          />
          <button className="btn btn-primary" disabled={sending} onClick={send}>
            {sending ? 'Sending…' : 'Send'}
          </button>
        </div>

        <div className="space-y-1.5 rounded-lg bg-black/20 p-3">
          <StatusLine label="planner" value={ros?.planner_status ?? null} />
          <StatusLine label="pick/place" value={ros?.pick_place_status ?? null} />
          <StatusLine label="arm" value={ros?.arm_status ?? null} />
        </div>

        {history.length > 0 && (
          <div>
            <div className="text-xs uppercase tracking-wider text-(--color-text-dim) font-mono mb-1">
              Recent commands
            </div>
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
