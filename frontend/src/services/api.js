const BASE = import.meta.env.VITE_BACKEND_URL || 'http://localhost:8000'
const WS_BASE = import.meta.env.VITE_WS_URL || 'ws://localhost:8000'

// ── REST helpers ──────────────────────────────────────────────
async function request(path, opts = {}) {
  const res = await fetch(`${BASE}${path}`, {
    headers: { 'Content-Type': 'application/json' },
    ...opts,
  })
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: res.statusText }))
    throw new Error(err.detail || 'Request failed')
  }
  return res.json()
}

// ── Health ────────────────────────────────────────────────────
export const getHealth  = () => request('/health')
export const getReady   = () => request('/ready')
export const getStatus  = () => request('/status')

// ── Session ───────────────────────────────────────────────────
export const startSession = (windowSize = 30, driverId = null) =>
  request('/session/start', {
    method: 'POST',
    body: JSON.stringify({ window_size: windowSize, driver_id: driverId }),
  })

export const stopSession = (sessionId) =>
  request('/session/stop', {
    method: 'POST',
    body: JSON.stringify({ session_id: sessionId }),
  })

export const getSessionStatus = (sessionId) =>
  request(`/session/${sessionId}/status`)

// ── Predict (REST, for testing) ───────────────────────────────
export const predict = (features, sessionId = null) =>
  request('/predict', {
    method: 'POST',
    body: JSON.stringify({ features, session_id: sessionId }),
  })

// ── WebSocket ─────────────────────────────────────────────────
export function createWebSocket(onMessage, onClose) {
  const ws = new WebSocket(WS_BASE)
  ws.onmessage = (e) => {
    try { onMessage(JSON.parse(e.data)) }
    catch { /* ignore malformed */ }
  }
  ws.onclose = onClose || (() => {})
  ws.onerror = (e) => console.error('WS error', e)
  return ws
}

// ── Airflow ───────────────────────────────────────────────────
const AIRFLOW_BASE = import.meta.env.VITE_AIRFLOW_URL || 'http://localhost:8080'

export async function getAirflowDagRuns(dagId = 'ddd_data_pipeline') {
  try {
    const res = await fetch(
      `${AIRFLOW_BASE}/api/v1/dags/${dagId}/dagRuns?limit=10&order_by=-execution_date`,
      { headers: { 'Authorization': 'Basic ' + btoa('admin:admin') } }
    )
    if (!res.ok) throw new Error('Airflow not reachable')
    return res.json()
  } catch {
    // Return mock data if Airflow not running
    return { dag_runs: MOCK_DAG_RUNS }
  }
}

export async function getAirflowTaskInstances(dagId, dagRunId) {
  try {
    const res = await fetch(
      `${AIRFLOW_BASE}/api/v1/dags/${dagId}/dagRuns/${dagRunId}/taskInstances`,
      { headers: { 'Authorization': 'Basic ' + btoa('admin:admin') } }
    )
    if (!res.ok) throw new Error()
    return res.json()
  } catch {
    return { task_instances: MOCK_TASK_INSTANCES }
  }
}

// ── Mock data (fallback when services not running) ────────────
const now = new Date()
const MOCK_DAG_RUNS = Array.from({ length: 5 }, (_, i) => ({
  dag_run_id: `manual__2026-04-${String(15 - i).padStart(2,'0')}`,
  dag_id: 'ddd_data_pipeline',
  execution_date: new Date(now - i * 86400000).toISOString(),
  state: i === 0 ? 'success' : i === 1 ? 'running' : 'success',
  start_date: new Date(now - i * 86400000 - 3600000).toISOString(),
  end_date: i === 1 ? null : new Date(now - i * 86400000).toISOString(),
}))

const MOCK_TASK_INSTANCES = [
  { task_id: 't0_validate_raw_data',  state: 'success', duration: 2.1 },
  { task_id: 't1_extract_frames',     state: 'success', duration: 182.4 },
  { task_id: 't2_extract_landmarks',  state: 'success', duration: 341.7 },
  { task_id: 't3_feature_engineering',state: 'success', duration: 54.2 },
  { task_id: 't4_split_data',         state: 'success', duration: 1.8 },
  { task_id: 't5_dvc_add_and_push',   state: 'success', duration: 8.3 },
  { task_id: 't6_pipeline_summary',   state: 'success', duration: 0.6 },
]
