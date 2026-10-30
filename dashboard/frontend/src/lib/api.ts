// Same-origin relative paths throughout -- vite.config.ts proxies /api to
// the backend in dev, and in a production build the backend can serve the
// built frontend itself so this stays same-origin there too. Never
// hardcode a backend host: that breaks the moment this is opened from a
// machine other than the one it was written on (this dashboard is
// explicitly meant to be reachable off the lab PC).
import type {
  ActionResult, CaptureResult, OutcomeSummary, ProcessInfo, StatusPayload,
} from './types'

export interface ArmLandmark {
  id: string; label: string; hint: string
  xyz: number[] | null
  clicked: [number, number] | null
  projected_now: [number, number] | null
}
export interface ArmCalibState {
  camera: string; frame_id: string; image: [number, number]
  arm_ok: boolean; k_ok: boolean; current_extrinsic_ok: boolean
  landmarks: ArmLandmark[]
  pool?: { n: number; rms_px: number; time: string }[]
}
export interface ArmCalibResult {
  ok?: boolean; error?: string
  rms_px?: number; warnings?: string[]
  landmarks_used?: string[]; reproj_px?: Record<string, number>; suspect?: string | null
  solved_projection?: Record<string, [number, number]>
  vs_current?: { rotation_deg: number; translation_m: number } | null
  pooled?: boolean; points?: number; worst_px?: number; mode?: string
  per_landmark_px?: Record<string, number>; offsets?: Record<string, number[]>
  poses?: { n: number; rms_px: number | null; single_pose_rms_px: number; time: string }[]
}

export interface JogState {
  enabled: boolean; message: string; moved: boolean; blocked: string | null; phrase: string
  speed_caps: { joint_deg_s: number; linear_cm_s: number; angular_deg_s: number }
  joint_age_s: number | null; joints_deg?: number[]; flange?: number[]
  lowest_link_above_limit_cm?: number; fault: boolean; arm_status: string; finger?: number
}

export interface LearningMap {
  label: string; labels: string[]; x: number[]; y: number[]
  grid: Record<string, [number, number][][]>          // method -> [x][y] -> [local success rate, evidence]
  side_allowed: boolean[][]
  dots: { method: string; x: number; y: number; ok: boolean }[]
  totals: Record<string, { trials: number; successes: number }>
  seed_zone: { x: number[]; y: number[] }
  params: { sigma_m: number; trust_radius_m: number; trust_min_rate: number; block_rate: number }
}

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
  outcomes: (liveOnly = true) => request<OutcomeSummary>(`/outcomes/summary?live_only=${liveOnly}`),
  learning: (label = 'cup') => request<LearningMap>(`/outcomes/learning?label=${encodeURIComponent(label)}`),
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

  armCalibState: (camera: string) => request<ArmCalibState>(`/arm_calib/state?camera=${camera}`),
  armCalibClick: (camera: string, landmark: string, u: number, v: number) =>
    request<{ ok: boolean }>('/arm_calib/click', { method: 'POST', body: JSON.stringify({ camera, landmark, u, v }) }),
  armCalibReset: (camera: string) => request<{ ok: boolean }>(`/arm_calib/reset?camera=${camera}`, { method: 'POST' }),
  armCalibSolve: (camera: string) => request<ArmCalibResult>(`/arm_calib/solve?camera=${camera}`, { method: 'POST' }),
  armCalibSave: (camera: string, pooled = false) =>
    request<{ ok: boolean; message: string; path: string | null }>(`/arm_calib/save_candidate?camera=${camera}${pooled ? '&pooled=true' : ''}`, { method: 'POST' }),
  armCalibCommit: (camera: string) =>
    request<{ ok: boolean; message: string; poses: number }>(`/arm_calib/commit_pose?camera=${camera}`, { method: 'POST' }),
  armCalibClearPool: (camera: string) => request<{ ok: boolean }>(`/arm_calib/clear_pool?camera=${camera}`, { method: 'POST' }),
  armCalibSolvePool: (camera: string, mode = 'all') => request<ArmCalibResult>(`/arm_calib/solve_pool?camera=${camera}&mode=${mode}`, { method: 'POST' }),
  jogState: () => request<JogState>('/jog/state'),
  jogEnable: (phrase: string) => request<{ ok: boolean; error?: string }>('/jog/enable', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ phrase }) }),
  jogDisable: () => request<{ ok: boolean }>('/jog/disable', { method: 'POST' }),
  jogStop: () => request<{ ok: boolean }>('/jog/stop', { method: 'POST' }),
  jogGripper: (action: 'open' | 'close' | 'set', position?: number) =>
    request<{ ok: boolean; message: string; finger?: number }>('/jog/gripper', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ action, position }) }),
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
