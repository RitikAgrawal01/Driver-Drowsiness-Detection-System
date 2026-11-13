"""
Stage 1: video -> per-frame landmarks at a true 10 fps.

Fixes vs v1:
  * applies each video's rotation metadata (v1 ignored it: 10 of 24 videos were sideways)
  * samples by timestamp, so 17 / 24 / 25 / 30 fps sources all give the same rate
  * keeps the real (rotated, resized) image size, so EAR/MAR/pose use correct pixel geometry
  * MediaPipe in video (tracking) mode per video, like the live backend
  * frames are streamed into MediaPipe; no JPEGs written to disk
  * frames with no face are KEPT (face=0) so later stages can measure face visibility

One DVC stage per RLDD fold (`extract_landmarks@<N>`), so adding a fold never re-processes
the others. The sealed test fold is refused.
Output: data/landmarks_10fps/fold<N>/<video>.csv, reports/extraction/fold<N>.json,
        reports/samples/fold<N>/<video>.jpg (visual check of rotation + landmarks)
"""

import argparse
import json
import multiprocessing as mp_proc
import time

import cv2
import pandas as pd

from ddd_core.config import load_params
from ddd_core.geometry import USED_LANDMARKS
from pipeline.dataset import LANDMARKS_DIR, REPORTS_DIR, list_videos, parse_video_name

SAMPLE_DIR = REPORTS_DIR / "samples"
ROTATIONS = {90: cv2.ROTATE_90_CLOCKWISE, 180: cv2.ROTATE_180, 270: cv2.ROTATE_90_COUNTERCLOCKWISE}


def process_video(args) -> dict:
    video_path, cfg, limit_s, out_dir, sample_dir = args
    import mediapipe as mp  # imported in the worker process

    name, participant, label = parse_video_name(video_path)
    cap = cv2.VideoCapture(str(video_path))
    cap.set(cv2.CAP_PROP_ORIENTATION_AUTO, 0)  # we rotate explicitly below
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    rotation = int(cap.get(cv2.CAP_PROP_ORIENTATION_META)) if cfg["apply_rotation"] else 0
    step = 1.0 / cfg["target_fps"]

    mesh = mp.solutions.face_mesh.FaceMesh(
        static_image_mode=False,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=cfg["min_detection_confidence"],
        min_tracking_confidence=cfg["min_tracking_confidence"],
    )

    rows, idx, next_t, sample_saved = [], -1, 0.0, False
    t0 = time.time()
    while True:
        if not cap.grab():
            break
        idx += 1
        t = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
        if t <= 0 and idx > 0:  # container without usable timestamps
            t = idx / src_fps
        if limit_s and t > limit_s:
            break
        if t + 1e-6 < next_t:
            continue
        while next_t <= t + 1e-6:
            next_t += step

        ok, frame = cap.retrieve()
        if not ok:
            continue
        if rotation in ROTATIONS:
            frame = cv2.rotate(frame, ROTATIONS[rotation])
        h, w = frame.shape[:2]
        scale = min(1.0, cfg["max_side"] / max(h, w))
        if scale < 1.0:
            frame = cv2.resize(frame, (round(w * scale), round(h * scale)), interpolation=cv2.INTER_AREA)
            h, w = frame.shape[:2]

        res = mesh.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        row = {
            "video": name,
            "participant": participant,
            "label": label,
            "t": round(t, 3),
            "width": w,
            "height": h,
            "face": 0,
        }
        if res.multi_face_landmarks:
            lm = res.multi_face_landmarks[0].landmark
            row["face"] = 1
            for i in USED_LANDMARKS:
                row[f"x{i}"] = lm[i].x * w
                row[f"y{i}"] = lm[i].y * h
            if not sample_saved and t >= 60:
                vis = frame.copy()
                for i in USED_LANDMARKS:
                    cv2.circle(vis, (int(row[f"x{i}"]), int(row[f"y{i}"])), 2, (0, 255, 0), -1)
                s = 480 / max(h, w)
                cv2.imwrite(str(sample_dir / f"{name}.jpg"), cv2.resize(vis, (round(w * s), round(h * s))))
                sample_saved = True
        rows.append(row)

    mesh.close()
    cap.release()
    df = pd.DataFrame(rows)
    df.to_csv(out_dir / f"{name}.csv", index=False)
    n, faces = len(df), int(df["face"].sum()) if len(df) else 0
    return {
        "video": name,
        "participant": participant,
        "label": label,
        "source_fps": round(src_fps, 2),
        "rotation_meta": rotation,
        "frame_size_after_rotation": [int(df["width"].iloc[0]), int(df["height"].iloc[0])] if n else None,
        "duration_s": float(df["t"].max()) if n else 0.0,
        "frames": n,
        "faces": faces,
        "detection_rate_pct": round(100 * faces / max(n, 1), 2),
        "elapsed_s": round(time.time() - t0, 1),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fold", type=int, required=True, help="RLDD fold to extract (data/raw/fold<N>)")
    ap.add_argument("--only", help="comma-separated video stems (for quick tests)")
    ap.add_argument("--limit-seconds", type=float, default=0, help="only the first N seconds (tests)")
    args = ap.parse_args()

    cfg = load_params()["extract"]
    out_dir, sample_dir = LANDMARKS_DIR / f"fold{args.fold}", SAMPLE_DIR / f"fold{args.fold}"
    out_dir.mkdir(parents=True, exist_ok=True)
    sample_dir.mkdir(parents=True, exist_ok=True)

    videos = list_videos(args.fold)  # refuses the sealed test fold
    if args.only:
        keep = set(args.only.split(","))
        videos = [v for v in videos if v.stem in keep]
    videos.sort(key=lambda p: p.stat().st_size, reverse=True)  # big ones first: better load balance

    t0 = time.time()
    ctx = mp_proc.get_context("spawn")
    with ctx.Pool(cfg["workers"]) as pool:
        results = []
        jobs = [(v, cfg, args.limit_seconds, out_dir, sample_dir) for v in videos]
        for r in pool.imap_unordered(process_video, jobs):
            print(
                f"{r['video']:>9}: {r['faces']}/{r['frames']} faces ({r['detection_rate_pct']}%), "
                f"rot {r['rotation_meta']}, {r['elapsed_s']} s",
                flush=True,
            )
            results.append(r)

    results.sort(key=lambda r: (r["participant"], r["label"]))
    frames = sum(r["frames"] for r in results)
    faces = sum(r["faces"] for r in results)
    summary = {
        "fold": args.fold,
        "target_fps": cfg["target_fps"],
        "videos": len(results),
        "frames": frames,
        "faces": faces,
        "detection_rate_pct": round(100 * faces / max(frames, 1), 2),
        "elapsed_s": round(time.time() - t0, 1),
        "per_video": results,
    }
    if not args.only and not args.limit_seconds:
        (REPORTS_DIR / "extraction").mkdir(parents=True, exist_ok=True)
        with open(REPORTS_DIR / "extraction" / f"fold{args.fold}.json", "w") as f:
            json.dump(summary, f, indent=2)
    print(json.dumps({k: v for k, v in summary.items() if k != "per_video"}, indent=2))


if __name__ == "__main__":
    main()
