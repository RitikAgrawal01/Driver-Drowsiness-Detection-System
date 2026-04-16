"""
airflow/scripts/download_dataset.py
────────────────────────────────────
Helper script to prepare the raw data directory.

DATASET OPTIONS (choose one):
─────────────────────────────
Option A — UTA Real-Life Drowsiness Dataset (RLDD)  ← RECOMMENDED
  • ~60 subjects, RGB video, labelled drowsy/alert/low-drowsy
  • Request access at: https://sites.google.com/view/utarldd/home
  • Place downloaded videos under:
      data/raw/drowsy/   (videos labelled as drowsy)
      data/raw/alert/    (videos labelled as alert)

Option B — Custom recordings (webcam)
  • Record yourself alert (eyes open, upright) for ~5 min
  • Record yourself drowsy (eyes closing, head nodding) for ~5 min
  • Place under data/raw/drowsy/ and data/raw/alert/

Option C — Quick-start synthetic test data  ← USE THIS FIRST
  • Run this script to generate synthetic landmark CSVs that bypass
    the video pipeline and let you test training immediately.
  • python airflow/scripts/download_dataset.py --synthetic

EXPECTED DIRECTORY STRUCTURE AFTER SETUP:
──────────────────────────────────────────
data/
├── raw/
│   ├── drowsy/
│   │   ├── subject_01_drowsy.mp4
│   │   └── ...
│   └── alert/
│       ├── subject_01_alert.mp4
│       └── ...
├── frames/          ← populated by extract_frames.py
├── landmarks/       ← populated by extract_landmarks.py
├── features/        ← populated by feature_engineering.py
└── processed/       ← populated by split_data.py

VIDEO FORMAT REQUIREMENTS:
───────────────────────────
• Format: .mp4 or .avi
• Resolution: at least 480×360 (720p recommended)
• Frame rate: at least 15 FPS (30 FPS recommended)
• Lighting: front-facing, face clearly visible
• Duration: at least 1 minute per clip recommended
"""

import argparse
import csv
import math
import os
import random
import sys

import numpy as np


def generate_synthetic_landmarks(output_dir: str, n_drowsy: int = 500, n_alert: int = 500):
    """
    Generate synthetic landmark CSV files to bootstrap the pipeline
    without real video data. Each row represents one frame.

    Landmark format: 468 landmarks × 3 coords (x, y, z) = 1404 columns
    plus a 'label' column (0=alert, 1=drowsy).

    Drowsy frames have:  lower EAR (more closed eyes), higher head pitch
    Alert  frames have:  higher EAR (open eyes),      neutral head pitch
    """
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "synthetic_landmarks.csv")

    # MediaPipe landmark indices used for EAR/MAR (right eye example)
    # We'll generate plausible 3D coords for all 468 landmarks
    # but make the key ones realistic for EAR computation

    print(f"Generating {n_drowsy} drowsy + {n_alert} alert synthetic frames...")

    header = []
    for i in range(468):
        header += [f"lm_{i}_x", f"lm_{i}_y", f"lm_{i}_z"]
    header.append("label")
    header.append("subject_id")
    header.append("frame_id")

    rows = []

    def _eye_landmarks(ear: float, center_x: float, center_y: float):
        """
        Generate 6 eye landmark coords that produce the given EAR.
        EAR = (||p2-p6|| + ||p3-p5||) / (2 × ||p1-p4||)
        We use a simplified model: horizontal width = 0.05, vertical = ear × horizontal
        """
        w = 0.05  # horizontal width
        v = ear * w  # vertical opening
        return [
            (center_x - w / 2, center_y, 0.0),   # p1 left corner
            (center_x - w / 4, center_y - v, 0.0),  # p2 upper-left
            (center_x + w / 4, center_y - v, 0.0),  # p3 upper-right
            (center_x + w / 2, center_y, 0.0),   # p4 right corner
            (center_x + w / 4, center_y + v, 0.0),  # p5 lower-right
            (center_x - w / 4, center_y + v, 0.0),  # p6 lower-left
        ]

    for label, n_frames in [(1, n_drowsy), (0, n_alert)]:
        for frame_id in range(n_frames):
            # Base landmark positions (random face-like positions)
            lm = {}
            for i in range(468):
                lm[i] = [
                    0.5 + random.gauss(0, 0.15),  # x centered ~0.5
                    0.5 + random.gauss(0, 0.15),  # y centered ~0.5
                    random.gauss(0, 0.01),         # z near 0
                ]

            # Override key eye landmarks with EAR-consistent values
            if label == 1:  # drowsy: EAR 0.10–0.22, noisy
                ear = random.gauss(0.16, 0.04)
                ear = max(0.05, min(0.30, ear))
            else:  # alert: EAR 0.25–0.40
                ear = random.gauss(0.32, 0.04)
                ear = max(0.22, min(0.45, ear))

            # Right eye landmarks (MediaPipe indices: 33,160,158,133,153,144)
            re = _eye_landmarks(ear, 0.65, 0.40)
            for idx, mp_idx in enumerate([33, 160, 158, 133, 153, 144]):
                lm[mp_idx] = list(re[idx])

            # Left eye landmarks (362,385,387,263,373,380)
            le = _eye_landmarks(ear, 0.35, 0.40)
            for idx, mp_idx in enumerate([362, 385, 387, 263, 373, 380]):
                lm[mp_idx] = list(le[idx])

            # Head pose noise (drowsy = more pitch)
            head_noise = random.gauss(-10 if label == 1 else -2, 3)

            row = []
            for i in range(468):
                row += lm[i]
            row.append(label)
            row.append(f"synthetic_{label}")
            row.append(frame_id)
            rows.append(row)

    # Shuffle rows
    random.shuffle(rows)

    with open(output_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)

    print(f"✓ Written {len(rows)} rows to {output_path}")
    print(f"  Drowsy frames: {n_drowsy},  Alert frames: {n_alert}")
    return output_path


def validate_raw_directory(raw_dir: str):
    """Check that the raw data directory has the expected structure."""
    issues = []
    drowsy_dir = os.path.join(raw_dir, "drowsy")
    alert_dir = os.path.join(raw_dir, "alert")

    if not os.path.isdir(drowsy_dir):
        issues.append(f"Missing directory: {drowsy_dir}")
    else:
        videos = [f for f in os.listdir(drowsy_dir) if f.endswith((".mp4", ".avi"))]
        if not videos:
            issues.append(f"No .mp4/.avi files found in {drowsy_dir}")
        else:
            print(f"✓ Found {len(videos)} drowsy video(s) in {drowsy_dir}")

    if not os.path.isdir(alert_dir):
        issues.append(f"Missing directory: {alert_dir}")
    else:
        videos = [f for f in os.listdir(alert_dir) if f.endswith((".mp4", ".avi"))]
        if not videos:
            issues.append(f"No .mp4/.avi files found in {alert_dir}")
        else:
            print(f"✓ Found {len(videos)} alert video(s) in {alert_dir}")

    if issues:
        print("\n⚠  Issues found:")
        for issue in issues:
            print(f"   • {issue}")
        return False
    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare DDD dataset")
    parser.add_argument(
        "--synthetic",
        action="store_true",
        help="Generate synthetic landmark data (no real videos needed)",
    )
    parser.add_argument(
        "--validate",
        action="store_true",
        help="Validate raw data directory structure",
    )
    parser.add_argument(
        "--data-dir",
        default="data",
        help="Root data directory (default: data/)",
    )
    parser.add_argument(
        "--n-drowsy",
        type=int,
        default=2000,
        help="Number of synthetic drowsy frames to generate",
    )
    parser.add_argument(
        "--n-alert",
        type=int,
        default=2000,
        help="Number of synthetic alert frames to generate",
    )
    args = parser.parse_args()

    if args.synthetic:
        out = generate_synthetic_landmarks(
            os.path.join(args.data_dir, "landmarks"),
            n_drowsy=args.n_drowsy,
            n_alert=args.n_alert,
        )
        print(f"\n✓ Synthetic data ready. Skip to feature_engineering.py.")
        print(f"  Run: python airflow/scripts/feature_engineering.py --input {out}")
    elif args.validate:
        ok = validate_raw_directory(os.path.join(args.data_dir, "raw"))
        sys.exit(0 if ok else 1)
    else:
        print(__doc__)
