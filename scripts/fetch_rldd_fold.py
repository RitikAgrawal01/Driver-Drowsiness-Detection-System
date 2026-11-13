"""
Download one UTA-RLDD fold's alert (KSS 0) and drowsy (KSS 10) videos from the Kaggle mirror.

The mirror was verified against the official Google Drive files: all 24 Fold 1 alert/drowsy
videos match byte-for-byte in size. Low-vigilance videos (5.*) are skipped.

  python scripts/fetch_rldd_fold.py --fold 2 --dest "/mnt/h/UTA RLDD Dataset/raw"

Writes <dest>/fold<N>/{alert,drowsy}/{alert,drowsy}<participant>.<ext> plus manifest.json
(source path, size, md5). Already-downloaded files with the right size are skipped, so the
script can be re-run after an interruption. Needs a Kaggle token in ~/.kaggle/access_token.
"""

import argparse
import csv
import hashlib
import io
import json
import re
import shutil
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path

DATASET = "rishab260/uta-reallife-drowsiness-dataset"
STATES = {"0": "alert", "10": "drowsy"}
KAGGLE = str(Path(sys.executable).with_name("kaggle"))


def list_files() -> list[dict]:
    out = subprocess.run(
        [KAGGLE, "datasets", "files", DATASET, "--page-size", "500", "--csv"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    return list(csv.DictReader(io.StringIO(out)))


def md5(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(16 << 20), b""):
            h.update(block)
    return h.hexdigest()


def download(name: str, size: int, target: Path, tries: int = 3) -> None:
    for attempt in range(1, tries + 1):
        with tempfile.TemporaryDirectory(dir=target.parent) as tmp:
            r = subprocess.run(
                [KAGGLE, "datasets", "download", DATASET, "-f", name, "-p", tmp], capture_output=True, text=True
            )
            got = [p for p in Path(tmp).rglob("*") if p.is_file()]  # Kaggle nests files in folders
            if r.returncode == 0 and got:
                f = got[0]
                if f.suffix == ".zip":  # Kaggle wraps each file in a zip
                    with zipfile.ZipFile(f) as z:
                        inner = z.infolist()[0]
                        z.extract(inner, tmp)
                    f.unlink()
                    f = Path(tmp) / inner.filename
                if f.stat().st_size == size:
                    shutil.move(str(f), target)
                    return
                print(f"  size mismatch on attempt {attempt}: {f.stat().st_size} != {size}", flush=True)
            else:
                print(f"  attempt {attempt} failed: {r.stderr.strip()[:200]}", flush=True)
    raise RuntimeError(f"could not download {name}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fold", type=int, required=True)
    ap.add_argument("--dest", required=True)
    args = ap.parse_args()

    root = Path(args.dest) / f"fold{args.fold}"
    pattern = re.compile(rf"Fold{args.fold}_part\d/[^/]+/(\d+)/(0|10)\.(\w+)$")
    wanted = []
    for row in list_files():
        m = pattern.match(row["name"])
        if m:
            participant, state, ext = int(m.group(1)), STATES[m.group(2)], m.group(3).lower()
            wanted.append((row["name"], int(row["size"]), root / state / f"{state}{participant}.{ext}"))
    if not wanted:
        sys.exit(f"no alert/drowsy files found for fold {args.fold}")
    total = sum(s for _, s, _ in wanted)
    print(f"fold {args.fold}: {len(wanted)} files, {total / 1e9:.1f} GB -> {root}", flush=True)

    manifest = []
    for i, (name, size, target) in enumerate(sorted(wanted, key=lambda w: str(w[2])), 1):
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists() and target.stat().st_size == size:
            print(f"[{i}/{len(wanted)}] {target.name}: already present", flush=True)
        else:
            print(f"[{i}/{len(wanted)}] {target.name} ({size / 1e9:.2f} GB) ...", flush=True)
            download(name, size, target)
        manifest.append(
            {
                "source": f"kaggle:{DATASET}/{name}",
                "file": str(target.relative_to(root)),
                "size": size,
                "md5": md5(target),
            }
        )
    (root / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"done: {len(manifest)} files verified by size; md5 in {root / 'manifest.json'}")


if __name__ == "__main__":
    main()
