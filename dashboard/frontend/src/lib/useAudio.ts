import { useEffect, useRef, useState } from 'react'
import type { AudioState } from './types'

const EMPTY: AudioState = { status: null, level: 0, mic_alive: false, transcript: null }

/** 10 Hz microphone state (level, status, latest transcript) over its own
 * WebSocket, kept separate from the 2 Hz status socket so the level meter
 * stays smooth. Only connects while a component using it is mounted. */
export function useAudio() {
  const [audio, setAudio] = useState<AudioState>(EMPTY)
  const retry = useRef(0)

  useEffect(() => {
    let ws: WebSocket
    let closedByUs = false
    let timer: ReturnType<typeof setTimeout>

    function connect() {
      const proto = window.location.protocol === 'https:' ? 'wss' : 'ws'
      ws = new WebSocket(`${proto}://${window.location.host}/api/ws/audio`)
      ws.onopen = () => { retry.current = 0 }
      ws.onmessage = (ev) => {
        try { setAudio(JSON.parse(ev.data)) } catch { /* ignore malformed frame */ }
      }
      ws.onclose = () => {
        setAudio(EMPTY)
        if (closedByUs) return
        timer = setTimeout(connect, Math.min(1000 * 2 ** retry.current++, 8000))
      }
      ws.onerror = () => ws.close()
    }
    connect()
    return () => { closedByUs = true; clearTimeout(timer); ws?.close() }
  }, [])

  return audio
}
