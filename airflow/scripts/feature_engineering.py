"""
airflow/scripts/feature_engineering.py
────────────────────────────────────────
Stage 3 of the DDD data pipeline.

Reads landmark CSV from data/landmarks/landmarks.csv
Uses PySpark to compute drowsiness features over a sliding window:
  • EAR  — Eye Aspect Ratio (mean, min, std per window)
  • PERCLOS — % of frames with EAR below threshold
  • MAR  — Mouth Aspect Ratio (mean, max per window)
  • Head Pose — pitch, yaw, roll angles (mean per window)

Also computes baseline statistics (mean, variance, distribution)
for all features → saved as data/features/baseline.json
(used later for drift detection in the backend)

Output:
  data/features/features.csv   — one row per sliding window
  data/features/baseline.json  — training distribution stats

Can be run:
  • Standalone:  python airflow/scripts/feature_engineering.py
  • By Airflow:  PythonOperator in dag_data_pipeline.py
  • By DVC:      dvc repro feature_engineering
"""

import json
import logging
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

# PySpark imports
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql import types as T
from pyspark.sql.window import Window

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
logger = logging.getLogger("feature_engineering")

# ── Config ────────────────────────────────────────────────────────────────────
LANDMARKS_CSV = os.getenv("LANDMARKS_CSV", "data/landmarks/landmarks.csv")
FEATURES_DIR = os.getenv("FEATURES_DIR", "data/features")
FEATURES_CSV = os.path.join(FEATURES_DIR, "features.csv")
BASELINE_JSON = os.path.join(FEATURES_DIR, "baseline.json")
WINDOW_SIZE = int(os.getenv("SLIDING_WINDOW_SIZE", "30"))
EAR_THRESHOLD = float(os.getenv("EAR_THRESHOLD", "0.25"))
MAR_THRESHOLD = float(os.getenv("MAR_THRESHOLD", "0.60"))

# ── MediaPipe Landmark Indices ────────────────────────────────────────────────
# Right eye: 33, 160, 158, 133, 153, 144
# Left eye:  362, 385, 387, 263, 373, 380
# Mouth outer: 61, 291, 39, 181, 0, 17, 269, 405  (simplified 8-point)
# Head pose reference: nose tip=1, chin=152, left eye corner=226, right eye corner=446,
#                      left mouth=57, right mouth=287
RIGHT_EYE = [33, 160, 158, 133, 153, 144]
LEFT_EYE = [362, 385, 387, 263, 373, 380]
MOUTH = [61, 291, 39, 181, 0, 17, 269, 405]
HEAD_POSE_PTS = [1, 152, 226, 446, 57, 287]  # for solvePnP reference


# ── Pure-Python feature functions (computed per frame before Spark) ───────────

def euclidean_2d(p1, p2):
    """2D Euclidean distance between two (x,y) tuples."""
    return math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)


def compute_ear(landmarks: dict, eye_indices: list) -> float:
    """
    Compute Eye Aspect Ratio for one eye.
    EAR = (||p2-p6|| + ||p3-p5||) / (2 * ||p1-p4||)
    """
    p = [(landmarks[f"lm_{i}_x"], landmarks[f"lm_{i}_y"]) for i in eye_indices]
    # p[0]=p1(left), p[1]=p2(upper-left), p[2]=p3(upper-right),
    # p[3]=p4(right), p[4]=p5(lower-right), p[5]=p6(lower-left)
    vertical_1 = euclidean_2d(p[1], p[5])
    vertical_2 = euclidean_2d(p[2], p[4])
    horizontal = euclidean_2d(p[0], p[3])
    if horizontal < 1e-6:
        return 0.0
    return (vertical_1 + vertical_2) / (2.0 * horizontal)


def compute_mar(landmarks: dict, mouth_indices: list) -> float:
    """
    Compute Mouth Aspect Ratio (simplified 4-point version).
    MAR = vertical_opening / horizontal_width
    Uses outer lip corners (idx 0,4) for horizontal,
    upper/lower lip mid-points (idx 2,6) for vertical.
    """
    if len(mouth_indices) < 8:
        return 0.0
    p = [(landmarks[f"lm_{i}_x"], landmarks[f"lm_{i}_y"]) for i in mouth_indices]
    # Horizontal: left corner (p[0]) to right corner (p[1])
    horizontal = euclidean_2d(p[0], p[1])
    # Vertical: upper lip (p[2]) to lower lip (p[6])
    vertical = euclidean_2d(p[2], p[6])
    if horizontal < 1e-6:
        return 0.0
    return vertical / horizontal


def compute_head_pose_angles(landmarks: dict) -> tuple:
    """
    Approximate head pose angles (pitch, yaw, roll) from 6 facial landmarks.
    Uses a simplified geometric approach:
      - Pitch:  vertical angle of nose-chin vector vs vertical axis
      - Yaw:    horizontal asymmetry of left/right eye corners
      - Roll:   angle of the eye line vs horizontal

    Returns (pitch_deg, yaw_deg, roll_deg)

    Note: For production accuracy, use cv2.solvePnP with a 3D face model.
    This approximation is sufficient for the sliding-window features.
    """
    try:
        # Key points (normalised coords in [0,1])
        nose_tip = (landmarks["lm_1_x"], landmarks["lm_1_y"])
        chin = (landmarks["lm_152_x"], landmarks["lm_152_y"])
        left_eye = (landmarks["lm_226_x"], landmarks["lm_226_y"])
        right_eye = (landmarks["lm_446_x"], landmarks["lm_446_y"])
        left_mouth = (landmarks["lm_57_x"], landmarks["lm_57_y"])
        right_mouth = (landmarks["lm_287_x"], landmarks["lm_287_y"])

        # Roll: angle of eye-line with horizontal
        eye_dx = right_eye[0] - left_eye[0]
        eye_dy = right_eye[1] - left_eye[1]
        roll_deg = math.degrees(math.atan2(eye_dy, eye_dx)) if abs(eye_dx) > 1e-6 else 0.0

        # Pitch: angle of nose-chin line vs vertical
        # (normalised: more negative = chin dropped = nodding off)
        nc_dx = chin[0] - nose_tip[0]
        nc_dy = chin[1] - nose_tip[1]
        nc_angle = math.degrees(math.atan2(nc_dx, nc_dy))
        pitch_deg = nc_angle  # positive = head back, negative = nodding forward

        # Yaw: asymmetry between left-eye-to-nose and right-eye-to-nose distances
        d_left = euclidean_2d(left_eye, nose_tip)
        d_right = euclidean_2d(right_eye, nose_tip)
        eye_span = euclidean_2d(left_eye, right_eye)
        yaw_deg = 0.0
        if eye_span > 1e-6:
            yaw_deg = math.degrees(math.asin(
                max(-1.0, min(1.0, (d_left - d_right) / eye_span))
            ))

        return round(pitch_deg, 4), round(yaw_deg, 4), round(roll_deg, 4)

    except (KeyError, ZeroDivisionError, ValueError):
        return 0.0, 0.0, 0.0


def compute_frame_features(row: dict) -> dict:
    """
    Compute all per-frame features from a landmark dict.
    Returns dict with: ear, mar, head_pitch, head_yaw, head_roll, label, subject_id, frame_id
    """
    right_ear = compute_ear(row, RIGHT_EYE)
    left_ear = compute_ear(row, LEFT_EYE)
    ear = (right_ear + left_ear) / 2.0  # average both eyes

    mar = compute_mar(row, MOUTH)
    pitch, yaw, roll = compute_head_pose_angles(row)

    return {
        "ear": round(ear, 6),
        "mar": round(mar, 6),
        "head_pitch": pitch,
        "head_yaw": yaw,
        "head_roll": roll,
        "label": int(row.get("label", 0)),
        "subject_id": str(row.get("subject_id", "unknown")),
        "frame_id": int(row.get("frame_id", 0)),
    }


# ── PySpark sliding window aggregation ───────────────────────────────────────

def create_spark_session() -> SparkSession:
    """Create a local PySpark session configured for the DDD pipeline."""
    return (
        SparkSession.builder
        .appName("DDD_FeatureEngineering")
        .master("local[*]")  # use all available local cores
        .config("spark.driver.memory", "2g")
        .config("spark.sql.shuffle.partitions", "8")
        .config("spark.sql.adaptive.enabled", "true")
        .getOrCreate()
    )


def apply_sliding_window_spark(frame_features_df: pd.DataFrame, window_size: int) -> pd.DataFrame:
    """
    Use PySpark to compute sliding-window aggregations over frame features.

    For each window of `window_size` consecutive frames per subject:
      - ear_mean, ear_min, ear_std
      - perclos (fraction of frames with EAR < EAR_THRESHOLD)
      - mar_mean, mar_max
      - head_pitch_mean, head_yaw_mean, head_roll_mean
      - window_label (majority vote of frame labels in window)

    Returns a pandas DataFrame with one row per window.
    """
    logger.info(f"Starting PySpark sliding window (window_size={window_size})...")
    spark = create_spark_session()

    # Convert pandas → Spark DataFrame
    sdf = spark.createDataFrame(frame_features_df)

    # Window spec: partition by subject, order by frame_id
    w = (
        Window
        .partitionBy("subject_id")
        .orderBy("frame_id")
        .rowsBetween(-(window_size - 1), 0)  # look back window_size-1 rows
    )

    # Compute window features
    sdf = sdf.withColumn("ear_mean", F.avg("ear").over(w))
    sdf = sdf.withColumn("ear_min", F.min("ear").over(w))
    sdf = sdf.withColumn("ear_std", F.stddev("ear").over(w))

    # PERCLOS: fraction of frames in window where EAR < threshold
    sdf = sdf.withColumn("eye_closed", (F.col("ear") < EAR_THRESHOLD).cast(T.IntegerType()))
    sdf = sdf.withColumn("perclos", F.avg("eye_closed").over(w))

    sdf = sdf.withColumn("mar_mean", F.avg("mar").over(w))
    sdf = sdf.withColumn("mar_max", F.max("mar").over(w))

    sdf = sdf.withColumn("head_pitch_mean", F.avg("head_pitch").over(w))
    sdf = sdf.withColumn("head_yaw_mean", F.avg("head_yaw").over(w))
    sdf = sdf.withColumn("head_roll_mean", F.avg("head_roll").over(w))

    # Window label: majority vote (avg > 0.5 → drowsy=1)
    sdf = sdf.withColumn("window_label", F.avg("label").over(w))
    sdf = sdf.withColumn(
        "label",
        (F.col("window_label") >= 0.5).cast(T.IntegerType())
    )

    # Select final feature columns
    feature_cols = [
        "subject_id", "frame_id",
        "ear_mean", "ear_min", "ear_std",
        "perclos",
        "mar_mean", "mar_max",
        "head_pitch_mean", "head_yaw_mean", "head_roll_mean",
        "label",
    ]
    sdf = sdf.select(*feature_cols)

    # Drop rows where we don't have a full window yet
    # (first window_size-1 frames per subject)
    sdf = sdf.dropna()

    # Collect back to pandas
    result_df = sdf.toPandas()
    spark.stop()

    logger.info(f"PySpark window aggregation complete: {len(result_df)} windows computed")
    return result_df


# ── Baseline statistics ───────────────────────────────────────────────────────

FEATURE_COLS = [
    "ear_mean", "ear_min", "ear_std",
    "perclos",
    "mar_mean", "mar_max",
    "head_pitch_mean", "head_yaw_mean", "head_roll_mean",
]


def compute_baseline_stats(features_df: pd.DataFrame) -> dict:
    """
    Compute training-time baseline statistics for drift detection.
    Saves mean and variance for each feature (Gaussian approximation).
    Also saves per-percentile histogram for KL-divergence computation.
    """
    baseline = {
        "description": "Training-time feature distribution baseline for drift detection",
        "window_size": WINDOW_SIZE,
        "n_samples": len(features_df),
        "features": {},
    }

    for col in FEATURE_COLS:
        if col not in features_df.columns:
            continue
        series = features_df[col].dropna()
        # Histogram bins for distribution approximation (20 bins)
        hist, bin_edges = np.histogram(series, bins=20, density=True)
        baseline["features"][col] = {
            "mean": round(float(series.mean()), 6),
            "std": round(float(series.std()), 6),
            "variance": round(float(series.var()), 6),
            "min": round(float(series.min()), 6),
            "max": round(float(series.max()), 6),
            "p5": round(float(series.quantile(0.05)), 6),
            "p25": round(float(series.quantile(0.25)), 6),
            "p50": round(float(series.quantile(0.50)), 6),
            "p75": round(float(series.quantile(0.75)), 6),
            "p95": round(float(series.quantile(0.95)), 6),
            # Store histogram as list for KL-divergence computation
            "histogram_density": [round(float(v), 6) for v in hist.tolist()],
            "histogram_bins": [round(float(v), 6) for v in bin_edges.tolist()],
        }

    logger.info(f"Baseline statistics computed for {len(baseline['features'])} features")
    return baseline


# ── Main pipeline function ────────────────────────────────────────────────────

def run_feature_engineering(
    landmarks_csv: str = LANDMARKS_CSV,
    features_dir: str = FEATURES_DIR,
    window_size: int = WINDOW_SIZE,
) -> dict:
    """
    Main feature engineering function.
    Reads landmarks.csv → computes per-frame features →
    applies PySpark sliding window → saves features.csv + baseline.json

    Returns summary dict for Airflow XCom.
    """
    os.makedirs(features_dir, exist_ok=True)

    summary = {
        "input_csv": landmarks_csv,
        "output_features_csv": FEATURES_CSV,
        "output_baseline_json": BASELINE_JSON,
        "window_size": window_size,
        "n_input_frames": 0,
        "n_output_windows": 0,
        "elapsed_sec": 0.0,
        "throughput_windows_per_sec": 0.0,
        "label_distribution": {},
    }

    start_time = time.time()

    # ── Step 1: Load landmark CSV ─────────────────────────────────────────────
    if not os.path.isfile(landmarks_csv):
        logger.error(f"Landmark CSV not found: {landmarks_csv}")
        logger.info("Tip: run extract_landmarks.py first, or use --synthetic mode")
        sys.exit(1)

    logger.info(f"Loading landmark CSV: {landmarks_csv}")
    lm_df = pd.read_csv(landmarks_csv)
    logger.info(f"Loaded {len(lm_df)} rows, {len(lm_df.columns)} columns")
    summary["n_input_frames"] = len(lm_df)

    # ── Step 2: Compute per-frame features ───────────────────────────────────
    logger.info("Computing per-frame features (EAR, MAR, head pose)...")
    frame_features = []
    for _, row in lm_df.iterrows():
        feat = compute_frame_features(row.to_dict())
        frame_features.append(feat)

    frame_df = pd.DataFrame(frame_features)
    logger.info(f"Per-frame features computed: {len(frame_df)} rows")

    # ── Step 3: PySpark sliding window ───────────────────────────────────────
    features_df = apply_sliding_window_spark(frame_df, window_size)

    # ── Step 4: Save features.csv ─────────────────────────────────────────────
    features_df.to_csv(FEATURES_CSV, index=False)
    logger.info(f"✓ Features saved: {FEATURES_CSV} ({len(features_df)} rows, {len(features_df.columns)} cols)")

    # ── Step 5: Compute and save baseline statistics ──────────────────────────
    baseline = compute_baseline_stats(features_df)
    with open(BASELINE_JSON, "w") as f:
        json.dump(baseline, f, indent=2)
    logger.info(f"✓ Baseline stats saved: {BASELINE_JSON}")

    elapsed = time.time() - start_time
    label_dist = features_df["label"].value_counts().to_dict()

    summary.update({
        "n_output_windows": len(features_df),
        "elapsed_sec": round(elapsed, 2),
        "throughput_windows_per_sec": round(len(features_df) / elapsed, 1) if elapsed > 0 else 0,
        "label_distribution": {str(k): int(v) for k, v in label_dist.items()},
    })

    logger.info(
        f"\n{'='*60}\n"
        f"FEATURE ENGINEERING COMPLETE\n"
        f"  Input frames   : {summary['n_input_frames']}\n"
        f"  Output windows : {summary['n_output_windows']}\n"
        f"  Label dist.    : {summary['label_distribution']}\n"
        f"  Throughput     : {summary['throughput_windows_per_sec']} windows/sec\n"
        f"  Elapsed        : {elapsed:.1f}s\n"
        f"{'='*60}"
    )

    return summary


def validate_output(features_csv: str, baseline_json: str) -> bool:
    """Validate that features.csv and baseline.json are well-formed."""
    passed = True

    # Check features.csv
    if not os.path.isfile(features_csv):
        logger.error(f"Validation failed: {features_csv} missing")
        return False

    df = pd.read_csv(features_csv)
    required_cols = FEATURE_COLS + ["label"]
    for col in required_cols:
        if col not in df.columns:
            logger.error(f"Validation failed: missing column '{col}' in features CSV")
            passed = False

    if df["label"].nunique() < 2:
        logger.error("Validation failed: only one class in features CSV")
        passed = False

    if passed:
        logger.info(f"✓ features.csv valid: {df.shape[0]} rows, label dist: {df['label'].value_counts().to_dict()}")

    # Check baseline.json
    if not os.path.isfile(baseline_json):
        logger.error(f"Validation failed: {baseline_json} missing")
        return False

    with open(baseline_json) as f:
        baseline = json.load(f)

    for col in FEATURE_COLS:
        if col not in baseline.get("features", {}):
            logger.error(f"Validation failed: feature '{col}' missing from baseline.json")
            passed = False

    if passed:
        logger.info(f"✓ baseline.json valid: {len(baseline['features'])} features")

    return passed


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default=LANDMARKS_CSV)
    parser.add_argument("--output-dir", default=FEATURES_DIR)
    parser.add_argument("--window-size", type=int, default=WINDOW_SIZE)
    args = parser.parse_args()

    summary = run_feature_engineering(
        landmarks_csv=args.input,
        features_dir=args.output_dir,
        window_size=args.window_size,
    )

    os.makedirs("reports", exist_ok=True)
    with open("reports/feature_engineering_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    ok = validate_output(FEATURES_CSV, BASELINE_JSON)
    sys.exit(0 if ok else 1)
