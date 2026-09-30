import { useEffect, useRef, useState } from 'react'
import { api } from '../lib/api'
import type { ProcessInfo } from '../lib/types'

export function SystemLog({ processes }: { processes: Record<string, ProcessInfo> }) {
  const ids = Object.keys(processes)
  const [selected, setSelected] = useState(ids[0] ?? '')
  const [log, setLog] = useState('')
  const boxRef = useRef<HTMLDivElement>(null)

  useEffect(() => {
    if (!selected && ids.length) setSelected(ids[0])
  }, [ids, selected])

  useEffect(() => {
    if (!selected) return
    let stop = false
    async function poll() {
      try {
        const res = await api.nodeLog(selected, 200)
        if (!stop) setLog(res.log)
      } catch {
        // node has no log yet (never started) -- keep showing empty
      }
      if (!stop) setTimeout(poll, 1500)
    }
    poll()
    return () => { stop = true }
  }, [selected])

  useEffect(() => {
    boxRef.current?.scrollTo({ top: boxRef.current.scrollHeight })
  }, [log])

  return (
    <div className="panel">
      <div className="panel-header">
        <h2 className="font-semibold text-sm tracking-tight">System Log</h2>
        <select
          className="bg-black/30 border border-(--color-border-bright) rounded-md px-2 py-1 text-xs font-mono outline-none"
          value={selected}
          onChange={(e) => setSelected(e.target.value)}
        >
          {ids.map((id) => (
            <option key={id} value={id}>{processes[id].label}</option>
          ))}
        </select>
      </div>
      <div
        ref={boxRef}
        className="p-3 h-56 overflow-y-auto text-[11px] font-mono leading-relaxed text-(--color-text-dim) whitespace-pre-wrap"
      >
        {log || 'no log output yet'}
      </div>
    </div>
  )
}
