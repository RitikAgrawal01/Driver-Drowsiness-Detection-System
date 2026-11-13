"""Create .env from .env.example with random secrets (keeps values already set in an existing .env).
Never prints secret values."""

import base64
import os
import re
import secrets
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SECRET_KEYS = {
    "ADMIN_TOKEN",
    "WEBHOOK_TOKEN",
    "AIRFLOW_ADMIN_PASSWORD",
    "AIRFLOW__WEBSERVER__SECRET_KEY",
    "GRAFANA_ADMIN_PASSWORD",
}


def existing(path: Path) -> dict:
    out = {}
    if path.exists():
        for line in path.read_text().splitlines():
            m = re.match(r"^([A-Z0-9_]+)=([^#\s]*)", line.strip())
            if m:
                out[m.group(1)] = m.group(2)
    return out


def main():
    old = existing(ROOT / ".env")
    lines = []
    for line in (ROOT / ".env.example").read_text().splitlines():
        m = re.match(r"^([A-Z0-9_]+)=([^#\s]*)", line)
        if not m:
            lines.append(line)
            continue
        key, val = m.group(1), m.group(2)
        if key in old and old[key] not in ("", "change-me"):
            val = old[key]
        elif key == "REPO_DIR":
            val = str(ROOT)
        elif key == "AIRFLOW__CORE__FERNET_KEY":
            val = base64.urlsafe_b64encode(os.urandom(32)).decode()
        elif key in SECRET_KEYS:
            val = secrets.token_urlsafe(24)
        elif key == "AIRFLOW_UID":
            val = str(os.getuid()) if hasattr(os, "getuid") else val
        lines.append(f"{key}={val}")
    (ROOT / ".env").write_text("\n".join(lines) + "\n")
    os.chmod(ROOT / ".env", 0o600)
    print("wrote .env (secrets not shown)")


if __name__ == "__main__":
    main()
