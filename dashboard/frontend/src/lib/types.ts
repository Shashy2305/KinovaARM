export type CalibrationStatus = 'ok' | 'needs_recalibration' | 'no_signal'

export interface SceneObject {
  label: string
  x: number
  y: number
  z: number | null
  confidence: number
  reachable: boolean
  stale: boolean
  relations?: string[]
}

export interface RosStatus {
  camera_calibration_status: Record<string, CalibrationStatus> | null
  scene_snapshot: Record<string, SceneObject> | null
  planner_status: string | null
  pick_place_status: string | null
  arm_status: string | null
  audio_status: string | null
  voice_transcript: Transcript | null
  joint_state_publishers?: number | null
}

export interface Transcript {
  text: string
  ts: number
  audio_s: number
  latency_s: number
}

export interface AudioState {
  status: string | null
  level: number
  mic_alive: boolean
  transcript: Transcript | null
}

export type ProcessStatus = 'stopped' | 'running' | 'running_external' | 'conflict'

export interface ProcessInfo {
  label: string
  category: 'robot' | 'cameras' | 'perception' | 'planning' | 'audio' | 'control'
  hardware_affecting: boolean
  signature: string
  conflict_signatures?: string[]
  status: ProcessStatus
}

export interface StatusPayload {
  ros: RosStatus
  processes: Record<string, ProcessInfo>
  events?: import('./pipeline').StatusEvent[]
}

export interface ActionResult {
  ok: boolean
  message: string
}

export interface CaptureResult {
  camera: string
  calib_file: string
  parent_frame: string
  child_frame: string
  translation: [number, number, number]
  rotation_quat: [number, number, number, number]
  restart_cmd: string
}
