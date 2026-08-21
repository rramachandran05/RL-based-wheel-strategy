"""Minimal .env loading. Search order: process env, project .env,
sibling ../wheel-strategy/.env (same keys, per SPEC-000 §7)."""
from __future__ import annotations

import os
from pathlib import Path

from rlbot.config import PROJECT_ROOT

_ENV_CANDIDATES = [
    PROJECT_ROOT / ".env",
    PROJECT_ROOT.parent / "wheel-strategy" / ".env",
]


def _parse_env_file(path: Path) -> dict:
    out = {}
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, val = line.partition("=")
        out[key.strip()] = val.strip().strip('"').strip("'")
    return out


def get_key(name: str) -> str | None:
    if os.getenv(name):
        return os.getenv(name)
    for candidate in _ENV_CANDIDATES:
        if candidate.exists():
            val = _parse_env_file(candidate).get(name)
            if val:
                return val
    return None
