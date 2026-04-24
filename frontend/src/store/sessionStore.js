// sessionStore.js
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

  // History (last 60 data points for charts)
  earHistory:        [],
  confidenceHistory: [],
  latencyHistory:    [],
  driftHistory:      [],

  // Alert log
  alertLog: [],

  // Drift
  overallDriftScore: 0,
  featureDriftScores: {},

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
        const ear = msg.features?.ear_mean ?? null;
        
        const newEarHistory = [...state.earHistory, { t: now, v: ear }].slice(-MAX);
        const newConfHistory = [...state.confidenceHistory, { t: now, v: msg.confidence }].slice(-MAX);
        const newLatHistory = [...state.latencyHistory, { t: now, v: msg.inference_latency_ms }].slice(-MAX);

        const isThisFrameDrowsy = msg.state === 'drowsy';
        
        let newDrowsyBuffer = isThisFrameDrowsy ? state.drowsyBuffer + 1 : 0;
        let newAlertBuffer = !isThisFrameDrowsy ? state.alertBuffer + 1 : 0;
        let nextState = state.currentState;

        // --- EDGE TRIGGER LOGIC ---
        let updatedAlertLog = state.alertLog;
        let updatedAlertsTriggered = state.alertsTriggered;

        // Check if we are TRANSITIONING to drowsy right now
        if (newDrowsyBuffer >= 15 && state.currentState !== 'drowsy') {
          nextState = 'drowsy';
          
          // Add ONE entry to the log exactly at the moment of transition
          const newEntry = {
            id: now,
            time: new Date().toLocaleTimeString(),
            confidence: msg.confidence,
            message: "Drowsiness Detected (State Change)",
          };
          updatedAlertLog = [newEntry, ...state.alertLog].slice(0, 20);
          updatedAlertsTriggered += 1;
        } 
        
        if (newAlertBuffer >= 10 && state.currentState !== 'alert') {
          nextState = 'alert';
        }

        set({
          currentState: nextState,
          drowsyBuffer: newDrowsyBuffer,
          alertBuffer: newAlertBuffer,
          currentConfidence: msg.confidence,
          currentEAR: ear,
          currentMAR: msg.features?.mar_mean ?? null,
          currentPERCLOS: msg.features?.perclos ?? null,
          framesProcessed: state.framesProcessed + 1,
          earHistory: newEarHistory,
          confidenceHistory: newConfHistory,
          latencyHistory: newLatHistory,
          // Update log and counter only if transition happened
          alertLog: updatedAlertLog,
          alertsTriggered: updatedAlertsTriggered,
        });
        break;
      }

      case 'alert': {
        break;
      }

      case 'drift_update': {
        set({
          overallDriftScore: msg.overall_drift_score,
          featureDriftScores: msg.feature_scores || {},
          driftHistory: [...state.driftHistory, { t: now, v: msg.overall_drift_score }].slice(-MAX)
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
    drowsyBuffer: 0,
    alertBuffer: 0,
  }),
}))


// ============================================================================
// FINAL MULTI-TAB SYNC LOGIC
// ============================================================================
const syncChannel = new BroadcastChannel('ddd_state_sync');

// 1. RECEIVER: Only update if we are NOT the one running the camera
syncChannel.onmessage = (event) => {
  const currentState = useSessionStore.getState();
  
  // If this tab doesn't have an active WebSocket, it should mirror the other tab
  if (!currentState.ws) {
    useSessionStore.setState(event.data);
  }
};

// 2. TRANSMITTER: Only the tab with the active camera broadcasts
useSessionStore.subscribe((state) => {
  // If this tab owns the active WebSocket, it is the "Source of Truth"
  if (state.ws && state.isActive) {
    const serializableState = Object.fromEntries(
      Object.entries(state).filter(([key, value]) => 
        typeof value !== 'function' && key !== 'ws'
      )
    );
    syncChannel.postMessage(serializableState);
  }
});