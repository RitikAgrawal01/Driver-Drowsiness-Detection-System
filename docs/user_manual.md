# User Manual
## Driver Drowsiness Detection System

**Version:** 1.0 | **Audience:** Non-technical users

---

## What Does This System Do?

The Driver Drowsiness Detection System watches your face through your webcam and alerts you if it detects signs of drowsiness — such as drooping eyelids, frequent blinking, or head nodding. The alert is both visual (red banner) and audio (warning tone).

---

## Quick Start (3 Steps)

### Step 1 — Open the Application
Open your web browser and go to: **http://localhost:3000**

You will see the **Live Monitor** page — this is your main screen.

### Step 2 — Start Monitoring
1. Click the **"START MONITORING"** button (cyan button at the bottom of the camera area)
2. Your browser will ask for permission to use your camera — click **"Allow"**
3. You will see your webcam feed appear with a **"CALIBRATING..."** label for the first 2 seconds while the system builds its baseline
4. Once calibration is done, the label changes to **"ALERT"** — the system is now watching

### Step 3 — Drive Safely
- The system runs in the background while you drive
- If you become drowsy, you will hear a **warning tone** and see a **red banner** at the top saying **"DROWSINESS DETECTED"**
- When you are alert again, the banner disappears automatically

### Stopping the Session
- Click **"STOP SESSION"** to end monitoring
- A summary will appear showing how long you drove and how many alerts were triggered

---

## Understanding the Gauges

The right side of the screen shows 4 real-time gauges:

| Gauge | What it measures | Danger zone |
|---|---|---|
| **EAR** (Eye Aspect Ratio) | How open your eyes are | Below 0.25 = eyes closing |
| **PERCLOS** | % of time your eyes were closed | Above 80% = drowsy |
| **Confidence** | How certain the AI is about its prediction | Above 70% = alert triggers |
| **MAR** (Mouth Aspect Ratio) | Whether you are yawning | Above 0.60 = yawning detected |

The **EAR chart** at the bottom left shows your eye openness over time. The red dashed line is the danger threshold — if your EAR drops below it frequently, you are likely drowsy.

---

## The Four Pages

Use the left sidebar to navigate between pages:

### 🔴 Live Monitor (Home)
Your main driving screen. Shows webcam, real-time gauges, and alert log.

### 🌿 Pipeline Console
Shows the data processing pipeline that was used to train the AI model. You can see each step (frame extraction, landmark detection, feature engineering) and whether they ran successfully. This is for technical review — you do not need to interact with it during driving.

### 📊 Monitoring Dashboard
Shows system performance charts: how fast the AI is making predictions, how confident it is, and whether the data it sees matches what it was trained on. If a **"DATA DRIFT DETECTED"** warning appears, it means conditions have changed significantly (e.g., very different lighting) and the system may be less accurate.

### 🤖 Model Registry
Shows the AI models that were trained and compared. The system automatically selected the best one (shown with a green **"Production"** badge) to use for detection.

---

## Tips for Best Accuracy

- **Lighting**: Make sure your face is well-lit from the front. Avoid driving into bright sunlight with the camera facing you.
- **Camera position**: The camera should be at eye level, roughly 30–60cm from your face.
- **Glasses**: Standard glasses are fine. Sunglasses may reduce accuracy — the system will show a drift warning if this happens.
- **Window size**: The system watches 30 frames at a time (~1 second). This means alerts trigger about 1 second after drowsiness begins.

---

## What the Colours Mean

| Colour | Meaning |
|---|---|
| 🔵 Cyan | System active and monitoring normally |
| 🟢 Green | Alert — driver is awake and driving safely |
| 🟡 Amber | Warning — watch carefully |
| 🔴 Red | Drowsiness detected — please pull over and rest |

---

## Troubleshooting

**"Camera access denied"**
→ Click the camera icon in your browser's address bar and select "Allow". Then click START MONITORING again.

**"Model server temporarily unavailable"**
→ The AI prediction service is starting up. Wait 10 seconds and try again. If it persists, contact your administrator.

**System shows "CALIBRATING" for a long time**
→ Make sure your face is clearly visible in the camera. The system needs to detect your face before it can start. Try adjusting your position or improving the lighting.

**No audio alert**
→ Check that your device volume is turned up. The sound alert button (top right of Live Monitor) should show "Sound On".

**The screen is blank / not loading**
→ Make sure the application is running. Open a terminal and run: `docker-compose up -d`

---

## Privacy

- Your webcam feed is **processed locally** on your device and is **never uploaded** to any external server
- No video is recorded or stored — only the mathematical features (eye ratios, head angles) are computed and used
- Session statistics (alert count, duration) are stored in memory only and cleared when you stop the session

---

*For technical documentation, see docs/HLD.md and docs/LLD.md*
