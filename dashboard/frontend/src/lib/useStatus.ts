import { useEffect, useRef, useState } from 'react'
import type { StatusPayload } from './types'

/** Live status over the backend's WS, with silent auto-reconnect -- a demo
 * shouldn't die because a WebSocket blipped once. */
export function useStatus() {
  const [status, setStatus] = useState<StatusPayload | null>(null)
  const [connected, setConnected] = useState(false)
  const retryRef = useRef(0)

  useEffect(() => {
    let ws: WebSocket
    let closedByUs = false
    let retryTimer: ReturnType<typeof setTimeout>

    function connect() {
      const proto = window.location.protocol === 'https:' ? 'wss' : 'ws'
      ws = new WebSocket(`${proto}://${window.location.host}/api/ws/status`)

      ws.onopen = () => {
        setConnected(true)
        retryRef.current = 0
      }
      ws.onmessage = (ev) => {
        try {
          setStatus(JSON.parse(ev.data))
        } catch {
          // ignore malformed frame
        }
      }
      ws.onclose = () => {
        setConnected(false)
        if (closedByUs) return
        const delay = Math.min(1000 * 2 ** retryRef.current, 8000)
        retryRef.current += 1
        retryTimer = setTimeout(connect, delay)
      }
      ws.onerror = () => ws.close()
    }

    connect()
    return () => {
      closedByUs = true
      clearTimeout(retryTimer)
      ws?.close()
    }
  }, [])

  return { status, connected }
}
