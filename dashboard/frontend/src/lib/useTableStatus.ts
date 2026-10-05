import { useEffect, useState } from 'react'
import { api, type TableStatus } from './api'

/** Whether the table geometry the safety guards depend on has been recorded.
 * Polled slowly -- it only changes when someone saves it. */
export function useTableStatus() {
  const [st, setSt] = useState<TableStatus | null>(null)
  useEffect(() => {
    let alive = true
    const tick = () => api.tableStatus().then((s) => alive && setSt(s)).catch(() => alive && setSt(null))
    tick()
    const id = setInterval(tick, 4000)
    return () => { alive = false; clearInterval(id) }
  }, [])
  return st
}
