// ============================================================================
// api.js — Backend communication layer (v2)
//
// * Every call goes to the backend. Airflow and MLflow are read through
//   allow-listed, read-only backend endpoints, so no credentials are shipped
//   in the browser bundle (v1 hard-coded Airflow admin:admin here).
// * Live monitoring is WebSocket-only: 'init' creates the session server-side.
// ============================================================================

const BASE = import.meta.env.VITE_BACKEND_URL || 'http://127.0.0.1:8000'
const WS_BASE = (import.meta.env.VITE_WS_URL || 'ws://127.0.0.1:8000').replace(/\/ws$/, '') + '/ws'

async function request(path, opts = {}, timeoutMs = 10000) {
  const controller = new AbortController()
  const timer = setTimeout(() => controller.abort(), timeoutMs)
  try {
    const res = await fetch(`${BASE}${path}`, {
      ...opts,
      signal: controller.signal,
      headers: { 'Content-Type': 'application/json', ...(opts.headers || {}) },
    })
    if (!res.ok) {
      const err = await res.json().catch(() => ({ detail: res.statusText }))
      throw new Error(err.detail || `HTTP ${res.status}`)
    }
    return res.json()
  } catch (e) {
    if (e.name === 'AbortError') throw new Error(`Backend timeout on ${path} after ${timeoutMs / 1000}s`)
    throw e
  } finally {
    clearTimeout(timer)
  }
}

// ── Health / model ───────────────────────────────────────────
export const getHealth = () => request('/health')
export const getReady = () => request('/ready')
export const getStatus = () => request('/status')
export const getModel = () => request('/api/model')
export const getDrift = () => request('/api/drift')

// ── Sessions (created by the WebSocket 'init') ──────────────
export const stopSession = (sessionId) => request(`/session/${sessionId}/stop`, { method: 'POST' })
export const getSessionStatus = (sessionId) => request(`/session/${sessionId}/status`)

// ── Predict (REST, for testing) ─────────────────────────────
export const predict = (features) => request('/predict', { method: 'POST', body: JSON.stringify({ features }) })

// ── WebSocket ───────────────────────────────────────────────
/** Resolves with an OPEN WebSocket, rejects on error/timeout. Handlers are attached before any event can fire. */
export function openWebSocket(onMessage, onClose, timeoutMs = 8000) {
  return new Promise((resolve, reject) => {
    let settled = false
    const ws = new WebSocket(WS_BASE)
    const timer = setTimeout(() => {
      if (settled) return
      settled = true
      reject(new Error('WebSocket timeout — backend unreachable at ' + WS_BASE))
    }, timeoutMs)
    ws.onmessage = (e) => {
      try { onMessage(JSON.parse(e.data)) } catch { /* ignore malformed JSON */ }
    }
    ws.onclose = onClose || (() => {})
    ws.onopen = () => {
      if (settled) return
      settled = true
      clearTimeout(timer)
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

// ── Airflow (read-only via backend) ─────────────────────────
export async function getAirflowDagRuns(dagId) {
  try {
    return await request(`/api/pipeline/${dagId}/runs?limit=10`)
  } catch (err) {
    console.error(`Failed to fetch DAG runs for ${dagId}:`, err)
    return { dag_runs: [] }
  }
}

export async function getAirflowTaskInstances(dagId, dagRunId) {
  try {
    return await request(`/api/pipeline/${dagId}/runs/${encodeURIComponent(dagRunId)}/tasks`)
  } catch (err) {
    console.error(`Failed to fetch tasks for ${dagRunId}:`, err)
    return { task_instances: [] }
  }
}

// ── MLflow (read-only via backend) ──────────────────────────
export async function getMLflowRuns() {
  try {
    const data = await request('/api/runs?limit=30')
    return (data.runs || []).map((r) => ({
      run_id: r.run_id,
      run_name: r.run_name || 'unnamed',
      model_type: r.params.model || 'unknown',
      feature_set: r.params.feature_set || '',
      status: r.status,
      start_time: new Date(r.start_time).toISOString(),
      metrics: r.metrics,
      params: r.params,
      stage: r.tags.stage || '',
    }))
  } catch (err) {
    console.error('Failed to fetch MLflow runs:', err)
    return []
  }
}
