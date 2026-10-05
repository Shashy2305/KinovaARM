import { useEffect, useRef, useState } from 'react'
import { api } from '../lib/api'
import type { AudioState } from '../lib/types'

const BARS = 28

type Mode = 'offline' | 'loading' | 'idle' | 'armed' | 'recording' | 'busy'

function modeOf(a: AudioState): Mode {
  if (!a.mic_alive && !a.status) return 'offline'
  const s = (a.status ?? '').toUpperCase()
  if (s.startsWith('LOADING')) return 'loading'
  if (s.startsWith('RECORDING')) return 'recording'
  if (s.startsWith('TRANSCRIBING')) return 'busy'
  if (s.startsWith('LISTENING')) return 'armed'
  if (!a.status) return 'offline'
  return 'idle'
}

function MicIcon() {
  return (
    <svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="square">
      <rect x="9" y="3" width="6" height="11" />
      <path d="M5 11a7 7 0 0 0 14 0M12 18v3M8 21h8" />
    </svg>
  )
}

export function MicButton({ audio }: { audio: AudioState }) {
  const mode = modeOf(audio)
  const [levels, setLevels] = useState<number[]>(() => Array(BARS).fill(0))
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const lastTick = useRef(0)

  // The meter is a scrolling history of the microphone RMS (~10 Hz).
  useEffect(() => {
    const now = performance.now()
    if (now - lastTick.current < 70) return
    lastTick.current = now
    setLevels((l) => [...l.slice(1), Math.min(1, Math.sqrt(audio.level) * 2.2)])
  }, [audio.level])

  async function call(action: Parameters<typeof api.audioControl>[0]) {
    setBusy(true)
    setError(null)
    try {
      await api.audioControl(action)
    } catch (e) {
      setError(String(e).replace(/^Error: \d+ /, ''))
    } finally {
      setBusy(false)
    }
  }

  const recording = mode === 'recording'
  const disabled = busy || mode === 'offline' || mode === 'loading' || mode === 'busy'
  const label: Record<Mode, string> = {
    offline: 'VOICE NODE OFFLINE',
    loading: 'LOADING SPEECH MODEL',
    idle: 'PRESS TO SPEAK',
    armed: 'HANDS-FREE · LISTENING',
    recording: 'RECORDING · PRESS TO SEND',
    busy: 'TRANSCRIBING',
  }

  return (
    <div className="flex items-stretch gap-3">
      <button
        type="button"
        aria-label={recording ? 'Stop recording' : 'Start recording'}
        aria-pressed={recording}
        disabled={disabled}
        onClick={() => call(recording ? 'stop' : 'start')}
        className={`w-14 shrink-0 border flex items-center justify-center transition-colors ${
          recording
            ? 'bg-(--color-red) border-(--color-red) text-black'
            : 'bg-(--color-panel-raised) border-(--color-border-bright) text-(--color-text) hover:border-(--color-amber)'
        } ${disabled ? 'opacity-40 cursor-not-allowed' : 'cursor-pointer'}`}
        style={{ borderRadius: 3 }}
      >
        <MicIcon />
      </button>

      <div className="flex-1 min-w-0 flex flex-col justify-between">
        <div className="flex items-center justify-between gap-2">
          <span className={`label ${recording ? '!text-(--color-red)' : mode === 'armed' ? '!text-(--color-green)' : ''}`}>
            {recording && <span className="led led-blink mr-1.5 align-middle" />}
            {label[mode]}
          </span>
          {mode !== 'offline' && mode !== 'loading' && (
            <label className="flex items-center gap-1.5 text-[11px] text-(--color-text-dim) cursor-pointer select-none">
              <input
                type="checkbox"
                className="accent-(--color-amber)"
                checked={mode === 'armed'}
                disabled={busy || mode === 'busy' || recording}
                onChange={(e) => call(e.target.checked ? 'arm' : 'disarm')}
              />
              hands-free
            </label>
          )}
          {mode === 'offline' && (
            <button className="btn !py-0.5 !text-[11px]" onClick={() => api.startNode('audio_node')}>
              Start voice node
            </button>
          )}
        </div>

        <div className="flex items-end gap-[2px] h-7" aria-hidden>
          {levels.map((v, i) => (
            <div
              key={i}
              className="flex-1"
              style={{
                height: `${Math.max(8, v * 100)}%`,
                background: recording ? 'var(--color-red)' : mode === 'armed' ? 'var(--color-green)' : '#3a4249',
                opacity: 0.35 + 0.65 * (i / BARS),
              }}
            />
          ))}
        </div>
        {error && <div className="text-[11px] text-(--color-red) mt-1">{error}</div>}
      </div>
    </div>
  )
}
