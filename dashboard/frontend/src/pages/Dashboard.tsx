import { useState } from 'react'
import { CalibrationWizard } from '../components/CalibrationWizard'
import { CommandConsole } from '../components/CommandConsole'
import { FusionView } from '../components/FusionView'
import { MultiCameraView } from '../components/MultiCameraView'
import { NodeControlGrid } from '../components/NodeControlGrid'
import { SceneView } from '../components/SceneView'
import { SystemLog } from '../components/SystemLog'
import { TopStatusBar } from '../components/TopStatusBar'
import { WorkspaceBoundary } from '../components/WorkspaceBoundary'
import { TableRecorder } from '../components/TableRecorder'
import { ArmCalibration } from '../components/ArmCalibration'
import { useStatus } from '../lib/useStatus'

const TABS = [
  { id: 'overview', label: 'Overview' },
  { id: 'fusion', label: '3D Fusion' },
  { id: 'calibration', label: 'Calibration' },
  { id: 'armcal', label: 'Calibrate from arm' },
  { id: 'boundary', label: 'Workspace boundary' },
  { id: 'table', label: 'Table safety' },
] as const

type TabId = (typeof TABS)[number]['id']

export function Dashboard() {
  const { status, connected, events } = useStatus()
  const [tab, setTab] = useState<TabId>('overview')

  const processes = status?.processes ?? {}
  // NodeControlGrid's actions mutate backend process state; the WS status
  // push (useStatus above) already reflects that on its own next tick, so
  // onRefresh here is a no-op hook kept for any future non-WS-driven view.
  const noopRefresh = () => {}

  return (
    <div className="min-h-screen flex">
      <aside className="w-52 shrink-0 border-r border-(--color-border) bg-(--color-panel) flex flex-col sticky top-0 h-screen">
        <div className="px-4 pt-5 pb-4 border-b border-(--color-border)">
          <div className="font-mono text-[0.95rem] font-semibold tracking-tight">KinovaARM</div>
          <div className="text-[11px] text-(--color-text-dim) mt-1 leading-snug">
            Gen3 7-DOF · Robotiq 2F-140<br />pick &amp; place by voice
          </div>
        </div>
        <nav className="flex-1 py-2">
          {TABS.map((t, i) => (
            <button
              key={t.id}
              onClick={() => setTab(t.id)}
              className={`w-full text-left flex items-baseline gap-2.5 px-4 py-2 text-[0.82rem] border-l-2 transition-colors ${
                tab === t.id
                  ? 'border-(--color-amber) text-(--color-text) bg-white/[0.03]'
                  : 'border-transparent text-(--color-text-dim) hover:text-(--color-text) hover:bg-white/[0.02]'
              }`}
            >
              <span className="font-mono text-[10px] text-(--color-text-faint)">{String(i + 1).padStart(2, '0')}</span>
              {t.label}
            </button>
          ))}
        </nav>
        <div className="px-4 py-3 border-t border-(--color-border) text-[10.5px] leading-snug text-(--color-text-faint)">
          <span className="text-(--color-amber)">No login.</span> Anyone on the lab network can reach this page and command the arm.
        </div>
      </aside>

      <div className="flex-1 min-w-0 flex flex-col">
        <TopStatusBar status={status} connected={connected} />

        <main className="flex-1 p-5 space-y-4">
          {tab === 'overview' && (
            <>
              <MultiCameraView />
              <div className="grid grid-cols-1 2xl:grid-cols-2 gap-4">
                <CommandConsole events={events} scene={status?.ros.scene_snapshot ?? null} />
                <SceneView scene={status?.ros.scene_snapshot ?? null} />
              </div>
              <NodeControlGrid processes={processes} onRefresh={noopRefresh} />
              <SystemLog processes={processes} />
            </>
          )}

          {tab === 'fusion' && <FusionView />}
          {tab === 'calibration' && <CalibrationWizard />}
          {tab === 'armcal' && <ArmCalibration />}
          {tab === 'boundary' && <WorkspaceBoundary />}
          {tab === 'table' && <TableRecorder />}
        </main>
      </div>
    </div>
  )
}
