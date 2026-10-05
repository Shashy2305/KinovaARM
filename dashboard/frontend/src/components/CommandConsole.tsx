import { useEffect, useMemo, useRef, useState } from 'react'
import { api } from '../lib/api'
import type { StatusEvent } from '../lib/pipeline'
import type { RosStatus } from '../lib/types'
import { useAudio } from '../lib/useAudio'
import { CommandPipeline } from './CommandPipeline'
import { MicButton } from './MicButton'

const BUILT_IN_SUGGESTIONS = ['go home', 'go left', 'go right']
const AUTO_KEY = 'kinova.voice.autosend'

function readAuto(): boolean {
  try { return localStorage.getItem(AUTO_KEY) === '1' } catch { return false }
}

export function CommandConsole({ events, scene }: { events: StatusEvent[]; scene: RosStatus['scene_snapshot'] }) {
  const audio = useAudio()
  const [text, setText] = useState('')
  const [sending, setSending] = useState(false)
  const [history, setHistory] = useState<string[]>([])
  const [autoSend, setAutoSend] = useState(readAuto)
  const [heard, setHeard] = useState('')            // transcript waiting for confirmation (editable)
  const seenTs = useRef<number | null>(null)

  // Objects the detector can actually see right now, not a fixed list.
  const objectSuggestions = useMemo(() => {
    if (!scene) return []
    const labels = new Set<string>()
    for (const obj of Object.values(scene)) if (obj.reachable && !obj.stale) labels.add(obj.label)
    return [...labels].slice(0, 6).map((l) => `pick up the ${l}`)
  }, [scene])

  async function sendText(t: string) {
    const cmd = t.trim()
    if (!cmd) return
    setSending(true)
    try {
      await api.sendCommand(cmd)
      setHistory((h) => [cmd, ...h].slice(0, 8))
    } finally {
      setSending(false)
    }
  }

  // A new transcript either waits for the operator's OK (default) or, with
  // auto-send on, goes straight to the planner. A transcript that was already
  // there when the page loaded is never replayed.
  useEffect(() => {
    const t = audio.transcript
    if (!t) return
    if (seenTs.current === null) { seenTs.current = t.ts; return }
    if (t.ts <= seenTs.current) return
    seenTs.current = t.ts
    if (autoSend) { sendText(t.text); setHeard('') } else setHeard(t.text)
  }, [audio.transcript]) // eslint-disable-line react-hooks/exhaustive-deps

  function toggleAuto(v: boolean) {
    setAutoSend(v)
    try { localStorage.setItem(AUTO_KEY, v ? '1' : '0') } catch { /* private mode */ }
  }

  return (
    <div className="panel">
      <div className="panel-header">
        <h2>Command console</h2>
        <label className="flex items-center gap-1.5 text-[11px] text-(--color-text-dim) cursor-pointer select-none">
          <input type="checkbox" className="accent-(--color-amber)" checked={autoSend} onChange={(e) => toggleAuto(e.target.checked)} />
          send voice commands without confirming
        </label>
      </div>
      <div className="p-4 space-y-4">
        <MicButton audio={audio} />

        {heard && (
          <div className="border border-(--color-amber) bg-(--color-amber)/[0.06] p-3 space-y-2">
            <div className="label !text-(--color-amber)">Heard — confirm before the arm moves</div>
            <input className="input font-mono" value={heard} onChange={(e) => setHeard(e.target.value)} />
            <div className="flex gap-2">
              <button className="btn btn-primary" disabled={sending || !heard.trim()} onClick={() => { sendText(heard); setHeard('') }}>
                Send to planner
              </button>
              <button className="btn" onClick={() => setHeard('')}>Discard</button>
            </div>
          </div>
        )}

        <div className="flex gap-2">
          <input
            className="input font-mono"
            value={text}
            onChange={(e) => setText(e.target.value)}
            onKeyDown={(e) => { if (e.key === 'Enter') { sendText(text); setText('') } }}
            placeholder="or type a command…"
            maxLength={300}
          />
          <button className="btn btn-primary shrink-0" disabled={sending || !text.trim()} onClick={() => { sendText(text); setText('') }}>
            {sending ? 'Sending' : 'Send'}
          </button>
        </div>

        <div className="flex flex-wrap gap-1.5">
          {BUILT_IN_SUGGESTIONS.map((s) => (
            <button key={s} className="suggestion-chip" onClick={() => sendText(s)}>{s}</button>
          ))}
          {objectSuggestions.map((s) => (
            <button key={s} className="suggestion-chip" onClick={() => sendText(s)}>{s}</button>
          ))}
          {objectSuggestions.length === 0 && (
            <span className="text-xs text-(--color-text-faint) py-1">no reachable objects detected yet</span>
          )}
        </div>

        <CommandPipeline events={events} />

        {history.length > 0 && (
          <div>
            <div className="label mb-1">Recent</div>
            {history.map((h, i) => (
              <div key={i} className="text-xs font-mono text-(--color-text-dim)">› {h}</div>
            ))}
          </div>
        )}
      </div>
    </div>
  )
}
