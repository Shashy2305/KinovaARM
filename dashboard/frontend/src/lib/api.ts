// Same-origin relative paths throughout -- vite.config.ts proxies /api to
// the backend in dev, and in a production build the backend can serve the
// built frontend itself so this stays same-origin there too. Never
// hardcode a backend host: that breaks the moment this is opened from a
// machine other than the one it was written on (this dashboard is
// explicitly meant to be reachable off the lab PC).
import type {
  ActionResult, CaptureResult, ProcessInfo, StatusPayload,
} from './types'

export interface TableStatus {
  pose: { x: number; y: number; z: number } | null
  pose_message: string
  points: number[][]
  saved: { table_top_z: number; x: number[]; y: number[]; rear_wall_x: number } | null
  saved_message: string | null
  flange_floor_z: number
}

const BASE = '/api'

async function request<T = unknown>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(`${BASE}${path}`, {
    headers: { 'Content-Type': 'application/json' },
    ...init,
  })
  if (!res.ok) {
    const text = await res.text().catch(() => res.statusText)
    throw new Error(`${res.status} ${text}`)
  }
  return res.json()
}

export const api = {
  status: () => request<StatusPayload>('/status'),
  nodes: () => request<Record<string, ProcessInfo>>('/nodes'),
  startNode: (id: string) => request<ActionResult>(`/nodes/${id}/start`, { method: 'POST' }),
  stopNode: (id: string) => request<ActionResult>(`/nodes/${id}/stop`, { method: 'POST' }),
  nodeLog: (id: string, lines = 200) => request<{ log: string }>(`/nodes/${id}/log?lines=${lines}`),
  fullBringup: () => request<{ ok: boolean; steps: { proc_id: string; ok: boolean; message: string }[] }>(
    '/bringup/full', { method: 'POST' }),
  armGoLive: (confirm_phrase: string) =>
    request<ActionResult>('/arm_controller/go_live', {
      method: 'POST',
      body: JSON.stringify({ confirm_phrase }),
    }),
  armGoDryRun: () => request<ActionResult>('/arm_controller/go_dry_run', { method: 'POST' }),

  generateBoard: () => request<ActionResult & { download_url: string }>(
    '/calibration/generate_board', { method: 'POST' }),
  boardVisible: (camera: string) =>
    request<{ wrist_ready: boolean; camera_ready: boolean; board_visible: boolean }>(
      `/calibration/board_visible/${camera}`),
  capture: (camera: string) =>
    request<{ ok: boolean; message: string; result: CaptureResult | null }>(
      `/calibration/capture/${camera}`, { method: 'POST' }),

  boundaryClick: (camera: string, u: number, v: number) =>
    request<{ ok: boolean; status: string; num_points: number }>('/workspace_boundary/click', {
      method: 'POST',
      body: JSON.stringify({ camera, u, v }),
    }),
  boundaryReset: (camera: string) =>
    request<ActionResult>(`/workspace_boundary/reset?camera=${camera}`, { method: 'POST' }),
  boundarySave: (camera: string) =>
    request<{ ok: boolean; status: string }>(`/workspace_boundary/save?camera=${camera}`, { method: 'POST' }),
  boundaryCurrent: () => request<Record<string, unknown>>('/workspace_boundary/current'),

  tableStatus: () => request<TableStatus>('/table_geometry/status'),
  tableRecord: () => request<{ ok: boolean; message: string }>('/table_geometry/record', { method: 'POST' }),
  tableReset: () => request<{ ok: boolean }>('/table_geometry/reset', { method: 'POST' }),
  tableSave: () => request<{ ok: boolean; message: string }>('/table_geometry/save', { method: 'POST' }),

  audioControl: (action: 'start' | 'stop' | 'cancel' | 'arm' | 'disarm') =>
    request<{ ok: boolean }>('/audio/control', { method: 'POST', body: JSON.stringify({ action }) }),

  sendCommand: (text: string) =>
    request<ActionResult>('/command', { method: 'POST', body: JSON.stringify({ text }) }),
}

export function cameraStreamUrl(camera: string) {
  return `${BASE}/camera/${camera}/stream`
}

export function boardPngUrl() {
  return `${BASE}/calibration/board.png?t=${Date.now()}`
}
