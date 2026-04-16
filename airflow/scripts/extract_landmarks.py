"""
airflow/scripts/extract_landmarks.py
──────────────────────────────────────
Stage 2 of the DDD data pipeline.

Reads extracted frames from data/frames/{drowsy,alert}/
Runs MediaPipe FaceMesh on each frame to extract 468 3D facial landmarks.
Saves per-frame landmark data as a single CSV: data/landmarks/landmarks.csv

Each row in the output CSV represents one frame with columns:
  frame_path, label, subject_id, frame_id,
  lm_0_x, lm_0_y, lm_0_z, lm_1_x, ... lm_467_z   (1404 landmark cols)

Frames where MediaPipe fails to detect a face are skipped and logged.
Throughput (frames/sec, detection rate) is logged and saved.

Can be run:
  • Standalone:  python airflow/scripts/extract_landmarks.py
  • By Airflow:  PythonOperator in dag_data_pipeline.py
  • By DVC:      dvc repro extract_landmarks
"""

import logging
import os
import re
import sys
import time
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
import pandas as pd

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("extract_landmarks")

# ── Config ────────────────────────────────────────────────────────────────────
FRAMES_DIR = os.getenv("FRAMES_DIR", "data/frames")
LANDMARKS_DIR = os.getenv("LANDMARKS_DIR", "data/landmarks")
OUTPUT_CSV = os.path.join(LANDMARKS_DIR, "landmarks.csv")
MIN_DETECTION_CONFIDENCE = float(os.getenv("MEDIAPIPE_MIN_DETECTION", "0.5"))
MIN_TRACKING_CONFIDENCE = float(os.getenv("MEDIAPIPE_MIN_TRACKING", "0.5"))
LABELS = ["drowsy", "alert"]
NUM_LANDMARKS = 468

# ── Build column names once ───────────────────────────────────────────────────
LANDMARK_COLS = []
for i in range(NUM_LANDMARKS):
    LANDMARK_COLS += [f"lm_{i}_x", f"lm_{i}_y", f"lm_{i}_z"]

META_COLS = ["frame_path", "label", "subject_id", "frame_id"]
ALL_COLS = META_COLS + LANDMARK_COLS


def _parse_subject_and_frame(filename: str, label: str) -> tuple:
    """
    Extract subject_id and frame_id from filename.
    Expected format: {label}_{video_stem}_f{frame_id:06d}.jpg
    Falls back gracefully for unexpected filenames.
    """
    match = re.search(r"_f(\d+)\.jpg$", filename)
    frame_id = int(match.group(1)) if match else 0

    # Subject is everything between label_ prefix and _f{id} suffix
    stem = Path(filename).stem
    subject_part = stem.replace(f"{label}_", "", 1)
    subject_part = re.sub(r"_f\d+$", "", subject_part)
    subject_id = subject_part if subject_part else "unknown"

    return subject_id, frame_id


def process_frame(
    frame_path: str,
    label: str,
    face_mesh,
) -> dict | None:
    """
    Run MediaPipe FaceMesh on a single frame.

    Args:
        frame_path: Path to JPEG frame.
        label:      Class label string ('drowsy' or 'alert').
        face_mesh:  Initialised mediapipe FaceMesh instance.

    Returns:
        dict with all META_COLS + LANDMARK_COLS, or None if detection failed.
    """
    img = cv2.imread(frame_path)
    if img is None:
        logger.warning(f"Cannot read frame: {frame_path}")
        return None

    # MediaPipe requires RGB
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(img_rgb)

    if not results.multi_face_landmarks:
        return None  # No face detected — skip frame

    # Use first detected face only
    face_landmarks = results.multi_face_landmarks[0]

    filename = Path(frame_path).name
    subject_id, frame_id = _parse_subject_and_frame(filename, label)

    row = {
        "frame_path": frame_path,
        "label": 1 if label == "drowsy" else 0,
        "subject_id": subject_id,
        "frame_id": frame_id,
    }

    # Extract normalised (x, y, z) for all 468 landmarks
    # MediaPipe normalises x,y to [0,1] relative to image dimensions
    # z is depth relative to nose tip (negative = closer to camera)
    for i, lm in enumerate(face_landmarks.landmark):
        row[f"lm_{i}_x"] = lm.x
        row[f"lm_{i}_y"] = lm.y
        row[f"lm_{i}_z"] = lm.z

    return row


def run_landmark_extraction(
    frames_dir: str = FRAMES_DIR,
    output_csv: str = OUTPUT_CSV,
    min_detection: float = MIN_DETECTION_CONFIDENCE,
    min_tracking: float = MIN_TRACKING_CONFIDENCE,
) -> dict:
    """
    Main landmark extraction function.
    Processes all frames for all labels and writes landmarks.csv.

    Returns summary dict (used as Airflow XCom).
    """
    os.makedirs(LANDMARKS_DIR, exist_ok=True)

    summary = {
        "total_frames_attempted": 0,
        "total_frames_succeeded": 0,
        "total_frames_no_face": 0,
        "detection_rate_pct": 0.0,
        "throughput_fps": 0.0,
        "elapsed_sec": 0.0,
        "per_label_counts": {},
        "output_csv": output_csv,
    }

    # Initialise MediaPipe FaceMesh
    # static_image_mode=True because we process individual frames (not video stream)
    # max_num_faces=1 — we track the driver only
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(
        static_image_mode=True,
        max_num_faces=1,
        refine_landmarks=True,  # enables iris landmarks (468 → 478 with iris)
        min_detection_confidence=min_detection,
        min_tracking_confidence=min_tracking,
    )

    all_rows = []
    pipeline_start = time.time()

    for label in LABELS:
        label_frames_dir = os.path.join(frames_dir, label)
        if not os.path.isdir(label_frames_dir):
            logger.warning(f"Frames directory not found: {label_frames_dir} — skipping")
            summary["per_label_counts"][label] = {"attempted": 0, "succeeded": 0}
            continue

        frame_files = sorted(Path(label_frames_dir).glob("*.jpg"))
        label_attempted = len(frame_files)
        label_succeeded = 0

        logger.info(f"Processing {label_attempted} frames for label '{label}'...")

        for i, frame_path in enumerate(frame_files):
            row = process_frame(str(frame_path), label, face_mesh)
            if row is not None:
                all_rows.append(row)
                label_succeeded += 1
            else:
                summary["total_frames_no_face"] += 1

            # Progress logging every 500 frames
            if (i + 1) % 500 == 0:
                logger.info(
                    f"  {label}: {i+1}/{label_attempted} frames processed "
                    f"({label_succeeded} faces detected)"
                )

        summary["total_frames_attempted"] += label_attempted
        summary["total_frames_succeeded"] += label_succeeded
        summary["per_label_counts"][label] = {
            "attempted": label_attempted,
            "succeeded": label_succeeded,
        }
        logger.info(
            f"✓ {label}: {label_succeeded}/{label_attempted} frames with face detected "
            f"({100*label_succeeded/max(label_attempted,1):.1f}%)"
        )

    face_mesh.close()
    elapsed = time.time() - pipeline_start

    if not all_rows:
        logger.error("No landmark rows extracted! Check that frames exist and faces are visible.")
        return summary

    # Build DataFrame and save
    df = pd.DataFrame(all_rows, columns=ALL_COLS)
    df.to_csv(output_csv, index=False)

    total = summary["total_frames_attempted"]
    succeeded = summary["total_frames_succeeded"]

    summary.update({
        "detection_rate_pct": round(100 * succeeded / max(total, 1), 2),
        "throughput_fps": round(total / elapsed, 1) if elapsed > 0 else 0,
        "elapsed_sec": round(elapsed, 2),
    })

    logger.info(
        f"\n{'='*60}\n"
        f"LANDMARK EXTRACTION COMPLETE\n"
        f"  Frames attempted  : {total}\n"
        f"  Faces detected    : {succeeded} ({summary['detection_rate_pct']}%)\n"
        f"  Frames skipped    : {summary['total_frames_no_face']}\n"
        f"  Throughput        : {summary['throughput_fps']} frames/sec\n"
        f"  Elapsed           : {elapsed:.1f}s\n"
        f"  Output CSV        : {output_csv}\n"
        f"  CSV shape         : {df.shape}\n"
        f"{'='*60}"
    )

    return summary


def validate_output(output_csv: str) -> bool:
    """Data quality check: ensure landmarks.csv has correct schema and both labels."""
    if not os.path.isfile(output_csv):
        logger.error(f"Validation failed: {output_csv} not found")
        return False

    df = pd.read_csv(output_csv, nrows=5)

    # Check required columns exist
    for col in META_COLS:
        if col not in df.columns:
            logger.error(f"Validation failed: missing column '{col}'")
            return False

    # Check landmark columns present (spot check first and last)
    for col in ["lm_0_x", "lm_467_z"]:
        if col not in df.columns:
            logger.error(f"Validation failed: missing landmark column '{col}'")
            return False

    # Check both labels present
    df_full = pd.read_csv(output_csv, usecols=["label"])
    label_counts = df_full["label"].value_counts().to_dict()
    logger.info(f"Label distribution: {label_counts}")

    if len(label_counts) < 2:
        logger.error("Validation failed: only one class label present in landmark CSV")
        return False

    logger.info(f"✓ Landmark CSV validation passed — shape: {df_full.shape[0]} rows")
    return True


if __name__ == "__main__":
    import json

    summary = run_landmark_extraction()

    os.makedirs("reports", exist_ok=True)
    with open("reports/landmark_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    ok = validate_output(OUTPUT_CSV)
    sys.exit(0 if ok else 1)
