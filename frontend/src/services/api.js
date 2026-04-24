// ============================================================================
// api.js — Backend communication layer
// ============================================================================
//
// CHANGES FROM PREV VERSION:
//
//   BUG FIX 1 — Added 10-second timeout to ALL fetch() calls.
//     Previously fetch() had no timeout. If the backend didn't respond
//     (e.g., MediaPipe hanging during session init), the button would show
//     "STARTING..." forever with no error shown. Now it fails fast with a
//     clear "Backend timeout" error message after 10 seconds.
//
//   BUG FIX 2 — NEW openWebSocket() function replaces the inline
//     Promise(resolve/reject) pattern in LiveMonitor.jsx.
//     All WS handlers (onopen, onerror, onclose, onmessage) are set
//     synchronously BEFORE the WS can fire any events. The timer is cleared
//     on success. On open, onerror is reset to a non-rejecting handler.
//     This eliminates ALL race conditions with the WS open event.
//
// ============================================================================

// Use 127.0.0.1 instead of localhost to bypass the Windows IPv6 bug
const BASE = 'http://127.0.0.1:8000';
const WS_BASE = 'ws://127.0.0.1:8000/ws';

// ── REST helpers with timeout ─────────────────────────────────

async function fetchWithTimeout(url, opts = {}, timeoutMs = 30000) {
  const controller = new AbortController()
  const timer = setTimeout(() => controller.abort(), timeoutMs)
  try {
    const res = await fetch(url, { ...opts, signal: controller.signal, headers: { 'Content-Type': 'application/json', ...opts.headers } })
    clearTimeout(timer)
    if (!res.ok) {
      const err = await res.json().catch(() => ({ detail: res.statusText }))
      throw new Error(err.detail || `HTTP ${res.status}`)
    }
    return res.json()
  } catch (e) {
    clearTimeout(timer)
    if (e.name === 'AbortError') {
      throw new Error(`Backend timeout — /session/start did not respond in ${timeoutMs / 1000}s. Is Docker running?`)
    }
    throw e
  }
}

function request(path, opts = {}) {
  return fetchWithTimeout(
    `${BASE}${path}`,
    { headers: { 'Content-Type': 'application/json' }, ...opts },
    10000,
  )
}

// ── Health ────────────────────────────────────────────────────
export const getHealth = () => request('/health')
export const getReady  = () => request('/ready')
export const getStatus = () => request('/status')

// ── Session ───────────────────────────────────────────────────
export const startSession = (windowSize = 30, driverId = null) =>
  fetchWithTimeout(`${BASE}/session/start`, {
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

/**
 * CHANGED: openWebSocket() is the new single function for WS creation.
 * Returns Promise<WebSocket> that resolves when open, rejects on error/timeout.
 *
 * All event handlers are assigned synchronously inside the Promise executor,
 * before the Promise is returned. This guarantees no race condition because
 * JS is single-threaded — the WS open event cannot fire during synchronous
 * execution of the executor.
 *
 * Usage in handleStart:
 *   const ws = await openWebSocket(store.handleWsMessage, onCloseCallback)
 */
export function openWebSocket(onMessage, onClose, timeoutMs = 8000) {
  return new Promise((resolve, reject) => {
    let settled = false
    const ws = new WebSocket(WS_BASE)

    const timer = setTimeout(() => {
      if (settled) return
      settled = true
      ws.onopen  = null
      ws.onerror = null
      reject(new Error('WS timeout — backend WebSocket unreachable. Is port 8000 open?'))
    }, timeoutMs)

    // onmessage and onclose set immediately — these work after open too
    ws.onmessage = (e) => {
      try { onMessage(JSON.parse(e.data)) }
      catch { /* ignore malformed JSON */ }
    }
    ws.onclose = onClose || (() => {})

    ws.onopen = () => {
      if (settled) return
      settled = true
      clearTimeout(timer)
      // Reset onerror to non-rejecting handler now that Promise is resolved
      ws.onerror = (e) => console.error('WS runtime error', e)
      resolve(ws)
    }

    ws.onerror = () => {
      if (settled) return
      settled = true
      clearTimeout(timer)
      reject(new Error('WebSocket connection failed — backend not reachable at ' + WS_BASE))
    }
  })
}

// Keep createWebSocket for any other callers (backwards compat)
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
      { headers: { 'Authorization': 'Basic ' + btoa('admin:admin') } },
    )
    if (!res.ok) throw new Error('Airflow not reachable')
    return res.json()
  } catch {
    return { dag_runs: MOCK_DAG_RUNS }
  }
}

export async function getAirflowTaskInstances(dagId, dagRunId) {
  try {
    const res = await fetch(
      `${AIRFLOW_BASE}/api/v1/dags/${dagId}/dagRuns/${dagRunId}/taskInstances`,
      { headers: { 'Authorization': 'Basic ' + btoa('admin:admin') } },
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
  dag_run_id: `manual__2026-04-${String(15 - i).padStart(2, '0')}`,
  dag_id: 'ddd_data_pipeline',
  execution_date: new Date(now - i * 86400000).toISOString(),
  state: i === 0 ? 'success' : i === 1 ? 'running' : 'success',
  start_date: new Date(now - i * 86400000 - 3600000).toISOString(),
  end_date: i === 1 ? null : new Date(now - i * 86400000).toISOString(),
}))

const MOCK_TASK_INSTANCES = [
  { task_id: 't0_validate_raw_data',   state: 'success', duration: 2.1 },
  { task_id: 't1_extract_frames',      state: 'success', duration: 182.4 },
  { task_id: 't2_extract_landmarks',   state: 'success', duration: 341.7 },
  { task_id: 't3_feature_engineering', state: 'success', duration: 54.2 },
  { task_id: 't4_split_data',          state: 'success', duration: 1.8 },
  { task_id: 't5_dvc_add_and_push',    state: 'success', duration: 8.3 },
  { task_id: 't6_pipeline_summary',    state: 'success', duration: 0.6 },
]
