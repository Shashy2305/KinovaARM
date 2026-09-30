import { useState } from 'react'
import { api } from '../lib/api'
import type { ProcessInfo } from '../lib/types'
import { StatusChip, toneForProcessStatus } from './StatusChip'

const CATEGORY_LABELS: Record<string, string> = {
  robot: 'Robot',
  cameras: 'Cameras',
  perception: 'Perception',
  planning: 'Planning',
  control: 'Control',
}

const CATEGORY_ORDER = ['robot', 'cameras', 'perception', 'planning', 'control']

export function NodeControlGrid({
  processes,
  onRefresh,
}: {
  processes: Record<string, ProcessInfo>
  onRefresh: () => void
}) {
  const [busy, setBusy] = useState<string | null>(null)
  const [lastMessage, setLastMessage] = useState<string | null>(null)
  const [bringupLog, setBringupLog] = useState<{ proc_id: string; ok: boolean; message: string }[] | null>(null)
  const [confirmLive, setConfirmLive] = useState(false)
  const [confirmText, setConfirmText] = useState('')

  async function run(id: string, action: 'start' | 'stop') {
    setBusy(id)
    try {
      const res = action === 'start' ? await api.startNode(id) : await api.stopNode(id)
      setLastMessage(res.message)
    } catch (e) {
      setLastMessage(String(e))
    } finally {
      setBusy(null)
      onRefresh()
    }
  }

  async function runFullBringup() {
    setBusy('__bringup__')
    setBringupLog(null)
    try {
      const res = await api.fullBringup()
      setBringupLog(res.steps)
    } catch (e) {
      setLastMessage(String(e))
    } finally {
      setBusy(null)
      onRefresh()
    }
  }

  async function submitGoLive() {
    setBusy('arm_controller')
    try {
      const res = await api.armGoLive(confirmText)
      setLastMessage(res.message)
      setConfirmLive(false)
      setConfirmText('')
    } catch (e) {
      setLastMessage(String(e))
    } finally {
      setBusy(null)
      onRefresh()
    }
  }

  async function goDryRun() {
    setBusy('arm_controller')
    try {
      const res = await api.armGoDryRun()
      setLastMessage(res.message)
    } catch (e) {
      setLastMessage(String(e))
    } finally {
      setBusy(null)
      onRefresh()
    }
  }

  const grouped = CATEGORY_ORDER.map((cat) => ({
    cat,
    items: Object.entries(processes).filter(([, p]) => p.category === cat),
  })).filter((g) => g.items.length > 0)

  return (
    <div className="panel">
      <div className="panel-header">
        <h2 className="font-semibold text-sm tracking-tight">Node Control</h2>
        <button
          className="btn btn-primary"
          disabled={busy === '__bringup__'}
          onClick={runFullBringup}
        >
          {busy === '__bringup__' ? 'Bringing up…' : '⚡ Full Bring-Up'}
        </button>
      </div>

      {bringupLog && (
        <div className="px-4 py-3 border-b border-(--color-border) bg-black/20 text-xs font-mono space-y-1">
          {bringupLog.map((s, i) => (
            <div key={i} className={s.ok ? 'text-(--color-green)' : 'text-(--color-red)'}>
              {s.ok ? '✓' : '✗'} {s.proc_id} — {s.message}
            </div>
          ))}
        </div>
      )}

      <div className="p-4 space-y-5">
        {grouped.map(({ cat, items }) => (
          <div key={cat}>
            <div className="text-xs uppercase tracking-wider text-(--color-text-dim) font-mono mb-2">
              {CATEGORY_LABELS[cat]}
            </div>
            <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-2">
              {items.map(([id, p]) => (
                <div
                  key={id}
                  className="rounded-lg border border-(--color-border) bg-(--color-panel-raised) px-3 py-2.5"
                >
                  <div className="flex items-center justify-between gap-2 mb-1.5">
                    <span className="text-sm font-medium leading-tight">{p.label}</span>
                    {p.hardware_affecting && (
                      <span title="Hardware-affecting" className="text-(--color-amber) text-xs">
                        ⚠
                      </span>
                    )}
                  </div>
                  <div className="flex items-center justify-between gap-2">
                    <StatusChip label={p.status.replace('_', ' ')} tone={toneForProcessStatus(p.status)} />
                    <div className="flex gap-1.5">
                      {id === 'arm_controller' ? (
                        <>
                          <button
                            className="btn"
                            disabled={busy === id}
                            onClick={goDryRun}
                          >
                            Dry-run
                          </button>
                          <button
                            className="btn btn-danger"
                            disabled={busy === id}
                            onClick={() => setConfirmLive(true)}
                          >
                            Go LIVE
                          </button>
                        </>
                      ) : (
                        <>
                          <button
                            className="btn"
                            disabled={busy === id || p.status === 'running' || p.status === 'running_external'}
                            onClick={() => run(id, 'start')}
                          >
                            Start
                          </button>
                          <button
                            className="btn btn-danger"
                            disabled={busy === id || p.status === 'stopped'}
                            onClick={() => run(id, 'stop')}
                          >
                            Stop
                          </button>
                        </>
                      )}
                    </div>
                  </div>
                </div>
              ))}
            </div>
          </div>
        ))}
      </div>

      {lastMessage && (
        <div className="px-4 py-2 border-t border-(--color-border) text-xs font-mono text-(--color-text-dim)">
          {lastMessage}
        </div>
      )}

      {confirmLive && (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/70 backdrop-blur-sm p-4">
          <div className="panel max-w-md w-full p-5 border-(--color-red)/50">
            <h3 className="text-(--color-red) font-semibold mb-2">⚠ Confirm LIVE arm control</h3>
            <p className="text-sm text-(--color-text-dim) mb-4">
              This restarts arm_controller with dry_run:=false. The arm will physically move on
              commanded actions. Make sure the workspace is clear and someone is on the E-stop.
              Type <span className="mono text-(--color-text)">MAKE IT LIVE</span> to confirm.
            </p>
            <input
              className="w-full bg-black/30 border border-(--color-border-bright) rounded-md px-3 py-2 mb-4 font-mono text-sm outline-none focus:border-(--color-red)"
              value={confirmText}
              onChange={(e) => setConfirmText(e.target.value)}
              placeholder="MAKE IT LIVE"
              autoFocus
            />
            <div className="flex justify-end gap-2">
              <button className="btn" onClick={() => { setConfirmLive(false); setConfirmText('') }}>
                Cancel
              </button>
              <button
                className="btn btn-danger"
                disabled={confirmText !== 'MAKE IT LIVE' || busy === 'arm_controller'}
                onClick={submitGoLive}
              >
                Confirm & Go Live
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  )
}
