"""The single source of truth for every tunable setting: params.yaml at the repo root.

Infrastructure (hosts, ports, passwords) lives in .env; model/feature/serving settings live
only in params.yaml. Containers mount the same file and point DDD_PARAMS at it.
"""

import os
from functools import lru_cache
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]


def params_path() -> Path:
    return Path(os.getenv("DDD_PARAMS", REPO_ROOT / "params.yaml"))


@lru_cache(maxsize=1)
def load_params() -> dict:
    with open(params_path()) as f:
        return yaml.safe_load(f)
