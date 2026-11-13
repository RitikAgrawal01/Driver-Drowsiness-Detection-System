"""
Fail if dvc.lock was produced with different params than params.yaml says (the v1 bug:
params.yaml/dvc.lock said window 2 while the reported model used window 30).

For every stage in every dvc.lock, each recorded parameter must equal the current value in
params.yaml. Run in CI (no data needed) and before committing results.
"""

import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]


def lookup(params: dict, dotted: str):
    node = params
    for part in dotted.split("."):
        node = node[part]
    return node


def flatten(prefix: str, value) -> dict:
    if isinstance(value, dict):
        out = {}
        for k, v in value.items():
            out.update(flatten(f"{prefix}.{k}" if prefix else k, v))
        return out
    return {prefix: value}


def check(root: Path = ROOT) -> list[str]:
    params = yaml.safe_load((root / "params.yaml").read_text())
    problems = []
    locks = [p for p in root.rglob("dvc.lock") if ".venv" not in p.parts and ".dvc" not in p.parts]
    for lock in locks:
        stages = yaml.safe_load(lock.read_text()).get("stages", {})
        for stage, body in stages.items():
            for _file, recorded in (body.get("params") or {}).items():
                for key, val in flatten("", recorded).items():
                    try:
                        current = lookup(params, key)
                    except (KeyError, TypeError):
                        problems.append(f"{lock.relative_to(root)}:{stage}: {key} no longer in params.yaml")
                        continue
                    if current != val:
                        problems.append(
                            f"{lock.relative_to(root)}:{stage}: {key} = {val!r} in lock, {current!r} in params.yaml"
                        )
    return problems


def main():
    problems = check()
    if problems:
        print("dvc.lock is out of sync with params.yaml (re-run `dvc repro` and commit dvc.lock):")
        print("\n".join(f"  - {p}" for p in problems))
        sys.exit(1)
    print("dvc.lock params match params.yaml")


if __name__ == "__main__":
    main()
