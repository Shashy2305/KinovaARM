import { useState } from 'react'
import { CalibrationWizard } from '../components/CalibrationWizard'
import { CommandConsole } from '../components/CommandConsole'
import { MultiCameraView } from '../components/MultiCameraView'
import { NodeControlGrid } from '../components/NodeControlGrid'
import { SceneView } from '../components/SceneView'
import { SystemLog } from '../components/SystemLog'
import { TopStatusBar } from '../components/TopStatusBar'
import { WorkspaceBoundary } from '../components/WorkspaceBoundary'
import { useStatus } from '../lib/useStatus'

const TABS = [
  { id: 'overview', label: 'Overview' },
  { id: 'calibration', label: 'Calibration' },
  { id: 'boundary', label: 'Workspace Boundary' },
] as const

type TabId = (typeof TABS)[number]['id']

export function Dashboard() {
  const { status, connected } = useStatus()
  const [tab, setTab] = useState<TabId>('overview')

  const processes = status?.processes ?? {}
  // NodeControlGrid's actions mutate backend process state; the WS status
  // push (useStatus above) already reflects that on its own next tick, so
  // onRefresh here is a no-op hook kept for any future non-WS-driven view.
  const noopRefresh = () => {}

  return (
    <div className="min-h-screen flex flex-col">
      <TopStatusBar status={status} connected={connected} />

      <nav className="flex gap-1 px-6 pt-4">
        {TABS.map((t) => (
          <button
            key={t.id}
            onClick={() => setTab(t.id)}
            className={`px-4 py-2 rounded-t-lg text-sm font-medium transition-colors border-b-2 ${
              tab === t.id
                ? 'border-(--color-cyan) text-(--color-cyan)'
                : 'border-transparent text-(--color-text-dim) hover:text-(--color-text)'
            }`}
          >
            {t.label}
          </button>
        ))}
      </nav>

      <main className="flex-1 p-6 pt-4 space-y-4">
        {tab === 'overview' && (
          <>
            <MultiCameraView />
            <div className="grid grid-cols-1 xl:grid-cols-2 gap-4">
              <SceneView scene={status?.ros.scene_snapshot ?? null} />
              <CommandConsole ros={status?.ros ?? null} />
            </div>
            <NodeControlGrid processes={processes} onRefresh={noopRefresh} />
            <SystemLog processes={processes} />
          </>
        )}

        {tab === 'calibration' && <CalibrationWizard />}

        {tab === 'boundary' && <WorkspaceBoundary />}
      </main>

      <footer className="px-6 py-3 text-[11px] text-(--color-text-faint) font-mono border-t border-(--color-border)">
        KinovaARM Dashboard · network-accessible, no auth — lab network only
      </footer>
    </div>
  )
}
