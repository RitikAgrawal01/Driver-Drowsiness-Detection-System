"""
airflow/scripts/extract_frames.py
──────────────────────────────────
Stage 1 of the DDD data pipeline.

Reads raw video files from data/raw/{drowsy,alert}/
Extracts frames at the configured FPS using OpenCV.
Saves frames as JPEG files to data/frames/{drowsy,alert}/
Logs throughput metrics (frames/sec, total time per video).

Can be run:
  • Standalone:  python airflow/scripts/extract_frames.py
  • By Airflow:  called via PythonOperator in dag_data_pipeline.py
  • By DVC:      dvc repro extract_frames
"""

import logging
import os
import sys
import time
from pathlib import Path

import cv2

# ── Logging setup ─────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger("extract_frames")

# ── Constants (overridden by params.yaml / env vars) ──────────────────────────
RAW_DIR = os.getenv("RAW_DATA_DIR", "data/raw")
FRAMES_DIR = os.getenv("FRAMES_DIR", "data/frames")
TARGET_FPS = int(os.getenv("FRAME_RATE", "30"))
SUPPORTED_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv"}

# Labels must match subdirectory names under data/raw/
LABELS = ["drowsy", "alert"]


def extract_frames_from_video(
    video_path: str,
    output_dir: str,
    target_fps: int,
    label: str,
) -> dict:
    """
    Extract frames from a single video file at target_fps.

    Args:
        video_path:  Path to the source video file.
        output_dir:  Directory to save extracted frame JPEGs.
        target_fps:  Target sampling rate (frames per second to extract).
        label:       Class label ('drowsy' or 'alert') for filename prefix.

    Returns:
        dict with keys: video_path, frames_extracted, duration_sec,
                         throughput_fps, elapsed_sec, status
    """
    result = {
        "video_path": video_path,
        "frames_extracted": 0,
        "duration_sec": 0.0,
        "throughput_fps": 0.0,
        "elapsed_sec": 0.0,
        "status": "failed",
    }

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        logger.error(f"Cannot open video: {video_path}")
        return result

    source_fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration_sec = total_frames / source_fps if source_fps > 0 else 0

    # Calculate frame sampling interval
    # e.g. source=30fps, target=15fps → take every 2nd frame
    frame_interval = max(1, round(source_fps / target_fps))

    logger.info(
        f"Processing {Path(video_path).name} | "
        f"source_fps={source_fps:.1f} target_fps={target_fps} "
        f"interval={frame_interval} total_frames={total_frames}"
    )

    os.makedirs(output_dir, exist_ok=True)
    video_stem = Path(video_path).stem

    start_time = time.time()
    frame_idx = 0
    saved_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Sample every frame_interval-th frame
        if frame_idx % frame_interval == 0:
            frame_filename = f"{label}_{video_stem}_f{frame_idx:06d}.jpg"
            frame_path = os.path.join(output_dir, frame_filename)
            cv2.imwrite(frame_path, frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
            saved_count += 1

        frame_idx += 1

    cap.release()
    elapsed = time.time() - start_time

    result.update({
        "frames_extracted": saved_count,
        "duration_sec": round(duration_sec, 2),
        "throughput_fps": round(saved_count / elapsed, 1) if elapsed > 0 else 0,
        "elapsed_sec": round(elapsed, 2),
        "status": "success",
    })

    logger.info(
        f"✓ {Path(video_path).name}: {saved_count} frames saved "
        f"in {elapsed:.1f}s ({result['throughput_fps']} frames/sec)"
    )
    return result


def run_extraction(
    raw_dir: str = RAW_DIR,
    frames_dir: str = FRAMES_DIR,
    target_fps: int = TARGET_FPS,
) -> dict:
    """
    Main extraction function. Processes all videos in raw_dir/{drowsy,alert}.

    Returns a summary dict with per-video stats and overall throughput.
    This dict is returned as an Airflow XCom value when called from a DAG.
    """
    summary = {
        "total_frames_extracted": 0,
        "total_videos_processed": 0,
        "total_elapsed_sec": 0.0,
        "overall_throughput_fps": 0.0,
        "per_video_stats": [],
        "errors": [],
    }

    pipeline_start = time.time()

    for label in LABELS:
        label_raw_dir = os.path.join(raw_dir, label)
        label_frames_dir = os.path.join(frames_dir, label)

        if not os.path.isdir(label_raw_dir):
            logger.warning(f"Raw directory not found: {label_raw_dir} — skipping")
            summary["errors"].append(f"Missing directory: {label_raw_dir}")
            continue

        video_files = [
            f for f in Path(label_raw_dir).iterdir()
            if f.suffix.lower() in SUPPORTED_EXTENSIONS
        ]

        if not video_files:
            logger.warning(f"No video files found in {label_raw_dir}")
            summary["errors"].append(f"No videos in {label_raw_dir}")
            continue

        logger.info(f"Found {len(video_files)} video(s) for label '{label}'")

        for video_path in sorted(video_files):
            stats = extract_frames_from_video(
                str(video_path),
                label_frames_dir,
                target_fps,
                label,
            )
            summary["per_video_stats"].append(stats)
            if stats["status"] == "success":
                summary["total_frames_extracted"] += stats["frames_extracted"]
                summary["total_videos_processed"] += 1
                summary["total_elapsed_sec"] += stats["elapsed_sec"]
            else:
                summary["errors"].append(f"Failed: {video_path}")

    pipeline_elapsed = time.time() - pipeline_start
    if pipeline_elapsed > 0:
        summary["overall_throughput_fps"] = round(
            summary["total_frames_extracted"] / pipeline_elapsed, 1
        )
    summary["total_elapsed_sec"] = round(pipeline_elapsed, 2)

    logger.info(
        f"\n{'='*60}\n"
        f"EXTRACTION COMPLETE\n"
        f"  Videos processed : {summary['total_videos_processed']}\n"
        f"  Frames extracted : {summary['total_frames_extracted']}\n"
        f"  Total time       : {summary['total_elapsed_sec']}s\n"
        f"  Throughput       : {summary['overall_throughput_fps']} frames/sec\n"
        f"  Errors           : {len(summary['errors'])}\n"
        f"{'='*60}"
    )

    if summary["errors"]:
        for err in summary["errors"]:
            logger.error(f"  ✗ {err}")

    return summary


def validate_output(frames_dir: str) -> bool:
    """
    Airflow data quality check: ensure frames directory has at least
    one frame for each label.
    """
    passed = True
    for label in LABELS:
        label_dir = os.path.join(frames_dir, label)
        if not os.path.isdir(label_dir):
            logger.error(f"Validation failed: {label_dir} not found")
            passed = False
            continue
        frames = list(Path(label_dir).glob("*.jpg"))
        if not frames:
            logger.error(f"Validation failed: no frames in {label_dir}")
            passed = False
        else:
            logger.info(f"✓ Validation passed: {len(frames)} frames in {label_dir}")
    return passed


if __name__ == "__main__":
    import json

    summary = run_extraction()

    # Save summary for DVC metrics and Airflow XCom
    os.makedirs("reports", exist_ok=True)
    summary_path = "reports/extraction_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    logger.info(f"Summary saved to {summary_path}")

    # Validate output
    ok = validate_output(FRAMES_DIR)
    if not ok:
        logger.error("Output validation failed!")
        sys.exit(1)

    sys.exit(0)
