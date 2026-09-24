"""Tiny .env loader so an API key can be supplied without exporting a shell
variable (and without the key ever appearing in a command line).

Create a file named ``.env`` in the project root containing, e.g.:

    DASHSCOPE_API_KEY=sk-xxxxxxxxxxxxxxxx

It is git-ignored. Values already present in the real environment win.
"""
from __future__ import annotations

import os
from pathlib import Path


def load_local_env(path: str | Path = ".env") -> list[str]:
    p = Path(path)
    if not p.exists():
        return []
    loaded = []
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, val = line.partition("=")
        key, val = key.strip(), val.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = val
            loaded.append(key)
    return loaded
