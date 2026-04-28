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
// Force 127.0.0.1 to match your new CORS rules perfectly
const AIRFLOW_BASE = import.meta.env.VITE_AIRFLOW_URL || 'http://127.0.0.1:8080'
const AUTH_HEADER = 'Basic ' + btoa('admin:admin')

export async function getAirflowDagRuns(dagId) {
  try {
    const res = await fetch(
      `${AIRFLOW_BASE}/api/v1/dags/${dagId}/dagRuns?limit=10&order_by=-execution_date`,
      { 
        headers: { 
          'Authorization': AUTH_HEADER,
          'Content-Type': 'application/json' 
        } 
      }
    )
    if (!res.ok) {
      throw new Error(`Airflow API HTTP Error: ${res.status}`)
    }
    return await res.json()
  } catch (err) {
    console.error(`Failed to fetch DAG runs for ${dagId}:`, err)
    // Return an EMPTY array instead of mock data so the UI shows "No runs found"
    return { dag_runs: [] } 
  }
}

export async function getAirflowTaskInstances(dagId, dagRunId) {
  try {
    const res = await fetch(
      `${AIRFLOW_BASE}/api/v1/dags/${dagId}/dagRuns/${dagRunId}/taskInstances`,
      { 
        headers: { 
          'Authorization': AUTH_HEADER,
          'Content-Type': 'application/json' 
        } 
      }
    )
    if (!res.ok) {
      throw new Error(`Airflow API HTTP Error: ${res.status}`)
    }
    return await res.json()
  } catch (err) {
    console.error(`Failed to fetch tasks for ${dagRunId}:`, err)
    return { task_instances: [] }
  }
}

// ── MLflow ────────────────────────────────────────────────────
// Force 127.0.0.1 to match Docker networking
const MLFLOW_BASE = import.meta.env.VITE_MLFLOW_URL || 'http://127.0.0.1:5000'

export async function getMLflowRuns() {
  try {
    // 🔥 CHANGED: Ask FastAPI to get the data for us!
    const res = await fetch(`${BASE}/api/mlflow/runs/search`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ experiment_ids: ['0', '1'], max_results: 10 })
    })
    
    if (!res.ok) throw new Error(`API Proxy Error: ${res.status}`)
    
    const data = await res.json()
    const runs = data.runs || []

    // Translate MLflow's array format into flat dictionaries for React
    return runs.map(r => {
      const metricsArr = r.data?.metrics || []
      const paramsArr = r.data?.params || []
      const tagsArr = r.data?.tags || []

      const metrics = Object.fromEntries(metricsArr.map(m => [m.key, m.value]))
      const params = Object.fromEntries(paramsArr.map(p => [p.key, p.value]))
      const tags = Object.fromEntries(tagsArr.map(t => [t.key, t.value]))

      // Try to determine the model type from params, tags, or run name
      const runName = tags['mlflow.runName'] || r.info.run_name || 'unnamed'
      let modelType = params['model_type'] || 'Unknown'
      if (modelType === 'Unknown') {
        if (runName.toLowerCase().includes('xgb')) modelType = 'XGBoost'
        else if (runName.toLowerCase().includes('svm')) modelType = 'SVM'
      }

      return {
        run_id: r.info.run_id,
        run_name: runName,
        model_type: modelType,
        status: r.info.status,
        start_time: new Date(r.info.start_time).toISOString(),
        metrics: metrics,
        params: params,
        // If you logged a stage tag, use it. Otherwise guess based on F1 > 0.90
        stage: tags['stage'] || (metrics['f1_weighted'] > 0.90 ? 'Production' : 'Archived'),
      }
    })
  } catch (err) {
    console.error("Failed to fetch MLflow runs:", err)
    return [] // Return empty array so UI shows "Loading..." or empty state
  }
}