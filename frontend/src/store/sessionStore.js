import { create } from 'zustand'

export const useSessionStore = create((set, get) => ({
  // Session
  sessionId: null,
  isActive: false,
  framesProcessed: 0,
  alertsTriggered: 0,

  // Current prediction
  currentState: 'idle',       // 'idle' | 'alert' | 'drowsy' | 'buffering'
  currentConfidence: 0,
  currentEAR: null,
  currentMAR: null,
  currentPERCLOS: null,

  // History (last 120 data points for charts)
  earHistory:        [],
  confidenceHistory: [],
  latencyHistory:    [],
  driftHistory:      [],

  // Alert log
  alertLog: [],

  // Drift
  overallDriftScore: 0,
  featureDriftScores: {},

  // KEPT for BroadcastChannel compatibility but no longer used for UI logic
  drowsyBuffer: 0,
  alertBuffer: 0,

  // WebSocket ref
  ws: null,

  // Actions
  setSession: (id) => set({ sessionId: id, isActive: true, framesProcessed: 0, alertsTriggered: 0 }),
  clearSession: () => set({ sessionId: null, isActive: false, currentState: 'idle' }),

  setWs: (ws) => set({ ws }),

  handleWsMessage: (msg) => {
    const state = get()
    const now = Date.now()
    const MAX = 120

    switch (msg.type) {

      case 'prediction': {
        const ear = msg.features?.ear_mean ?? null

        const newEarHistory  = [...state.earHistory,        { t: now, v: ear }].slice(-MAX)
        const newConfHistory = [...state.confidenceHistory, { t: now, v: msg.confidence }].slice(-MAX)
        const newLatHistory  = [...state.latencyHistory,    { t: now, v: msg.inference_latency_ms }].slice(-MAX)

        let updatedAlertLog       = state.alertLog
        let updatedAlertsTriggered = state.alertsTriggered

        if (msg.new_alert) {
          // CHANGED: was triggered by frontend re-buffer reaching 15.
          // Now triggered by backend's new_alert flag (single edge event).
          const newEntry = {
            id: now,
            time: new Date().toLocaleTimeString(),
            confidence: msg.confidence,
            message: 'Drowsiness Detected',
          }
          updatedAlertLog       = [newEntry, ...state.alertLog].slice(0, 20)
          updatedAlertsTriggered = state.alertsTriggered + 1
        }

        set({
          // CHANGED: was 'nextState' computed from frontend buffers.
          // Now directly uses msg.state from backend.
          currentState:      msg.state,
          currentConfidence: msg.confidence,
          currentEAR:        ear,
          currentMAR:        msg.features?.mar_mean  ?? null,
          currentPERCLOS:    msg.features?.perclos   ?? null,
          framesProcessed:   state.framesProcessed + 1,
          earHistory:        newEarHistory,
          confidenceHistory: newConfHistory,
          latencyHistory:    newLatHistory,
          alertLog:          updatedAlertLog,
          alertsTriggered:   updatedAlertsTriggered,
        })
        break
      }

      // 'alert' messages from backend are currently informational only
      case 'alert': {
        break
      }

      case 'drift_update': {
        set({
          overallDriftScore:  msg.overall_drift_score,
          featureDriftScores: msg.feature_scores || {},
          driftHistory: [...state.driftHistory, { t: now, v: msg.overall_drift_score }].slice(-MAX),
        })
        break
      }

      case 'buffering': {
        set({ currentState: 'buffering' })
        break
      }

      default: break
    }
  },

  reset: () => set({
    earHistory: [], confidenceHistory: [], latencyHistory: [],
    driftHistory: [], alertLog: [], framesProcessed: 0,
    alertsTriggered: 0, currentState: 'idle', overallDriftScore: 0,
    // KEPT for compat — reset to zero
    drowsyBuffer: 0,
    alertBuffer: 0,
  }),
}))


// ============================================================================
// MULTI-TAB SYNC via BroadcastChannel
// Only the tab with the active WebSocket broadcasts; others mirror it.
// ============================================================================
const syncChannel = new BroadcastChannel('ddd_state_sync')

syncChannel.onmessage = (event) => {
  const currentState = useSessionStore.getState()
  if (!currentState.ws) {
    useSessionStore.setState(event.data)
  }
}

useSessionStore.subscribe((state) => {
  if (state.ws && state.isActive) {
    const serializableState = Object.fromEntries(
      Object.entries(state).filter(([key, value]) =>
        typeof value !== 'function' && key !== 'ws'
      )
    )
    syncChannel.postMessage(serializableState)
  }
})
