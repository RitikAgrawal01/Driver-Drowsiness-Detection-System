import { create } from 'zustand'

// States sent by the backend (one 'status' message per frame):
//   warming_up | alert | drowsy | no_face | degraded | error  (+ 'idle' locally)
const MAX = 120

export const useSessionStore = create((set, get) => ({
  sessionId: null,
  isActive: false,
  windowS: null,
  targetFps: 10,
  modelVersion: null,

  framesProcessed: 0,
  alertsTriggered: 0,
  currentState: 'idle',
  alarm: false,
  warmup: 0,
  fps: 0,
  secondsWithoutFace: 0,
  lastError: null,

  currentScore: null,
  threshold: null,
  currentEAR: null,
  currentMAR: null,
  currentPERCLOS: null,
  blinkRate: null,

  earHistory: [],
  confidenceHistory: [],   // model score history (name kept for the dashboard)
  latencyHistory: [],
  driftHistory: [],
  alertLog: [],

  overallDriftScore: -1,
  featureDriftScores: {},
  driftThreshold: null,   // from the backend (params.yaml drift.psi_threshold)

  ws: null,

  setWs: (ws) => set({ ws }),
  clearSession: () => set({ sessionId: null, isActive: false, currentState: 'idle', alarm: false }),

  setDrift: (d) => {
    if (!d || d.overall === undefined) return
    const now = Date.now()
    set((s) => ({
      driftThreshold: d.threshold ?? s.driftThreshold,
      overallDriftScore: d.overall,
      featureDriftScores: d.features || {},
      driftHistory: d.overall >= 0 ? [...s.driftHistory, { t: now, v: d.overall }].slice(-MAX) : s.driftHistory,
    }))
  },

  handleWsMessage: (msg) => {
    const s = get()
    const now = Date.now()
    switch (msg.type) {
      case 'init_ack':
        set({ sessionId: msg.session_id, isActive: true, windowS: msg.window_s,
              targetFps: msg.target_fps, modelVersion: msg.model_version, currentState: 'warming_up' })
        break
      case 'status': {
        const f = msg.features || {}
        const patch = {
          currentState: msg.state, alarm: msg.alarm, warmup: msg.warmup, fps: msg.fps,
          secondsWithoutFace: msg.seconds_without_face, lastError: msg.error || null,
          framesProcessed: s.framesProcessed + (msg.processed ? 1 : 0),
        }
        if (msg.predicted) {
          patch.currentScore = msg.score
          patch.threshold = msg.threshold
          patch.currentEAR = f.ear_mean ?? null
          patch.currentMAR = f.mar_mean ?? null
          patch.currentPERCLOS = f.perclos ?? null
          patch.blinkRate = f.blink_rate_pm ?? null
          patch.earHistory = [...s.earHistory, { t: now, v: f.ear_mean }].slice(-MAX)
          patch.confidenceHistory = [...s.confidenceHistory, { t: now, v: msg.score }].slice(-MAX)
        }
        if (msg.new_alarm) {
          patch.alertLog = [{ id: now, time: new Date().toLocaleTimeString(), score: msg.score }, ...s.alertLog].slice(0, 20)
          patch.alertsTriggered = s.alertsTriggered + 1
        }
        set(patch)
        break
      }
      case 'error':
        set({ lastError: msg.message, ...(msg.state ? { currentState: msg.state } : {}) })
        break
      default:
        break
    }
  },

  recordLatency: (ms) => set((s) => ({ latencyHistory: [...s.latencyHistory, { t: Date.now(), v: ms }].slice(-MAX) })),

  reset: () => set({
    earHistory: [], confidenceHistory: [], latencyHistory: [], driftHistory: [], alertLog: [],
    framesProcessed: 0, alertsTriggered: 0, currentState: 'idle', alarm: false, overallDriftScore: -1,
    currentScore: null, currentEAR: null, currentMAR: null, currentPERCLOS: null, lastError: null,
  }),
}))

// Multi-tab mirror: only the tab holding the WebSocket broadcasts; other tabs display it.
const syncChannel = typeof BroadcastChannel !== 'undefined' ? new BroadcastChannel('ddd_state_sync') : null
if (syncChannel) {
  syncChannel.onmessage = (event) => {
    if (!useSessionStore.getState().ws) useSessionStore.setState(event.data)
  }
  useSessionStore.subscribe((state) => {
    if (state.ws && state.isActive) {
      syncChannel.postMessage(Object.fromEntries(
        Object.entries(state).filter(([k, v]) => typeof v !== 'function' && k !== 'ws')))
    }
  })
}
