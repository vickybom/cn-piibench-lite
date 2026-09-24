#!/usr/bin/env python
"""Check that the API key + base URL in .env work, and discover which Qwen-2.5
model IDs are actually served on your account.

Usage:
    python test_connection.py                 # probe a default candidate list
    python test_connection.py qwen2.5-7b-instruct qwen-plus   # probe specific ids

The key is read from .env (DASHSCOPE_API_KEY) and never printed.
"""
from __future__ import annotations

import sys
from pii_auditor.localenv import load_local_env
from pii_auditor.m3_inference import OpenAICompatBackend

CANDIDATES = [
    "qwen2.5-1.5b-instruct",
    "qwen2.5-3b-instruct",
    "qwen2.5-7b-instruct",
    "qwen2.5-14b-instruct",
    "qwen2.5-32b-instruct",
    "qwen2.5-72b-instruct",
    "qwen-plus",
    "qwen-turbo",
]

PROBE = {
    "pid": 0, "category": "full_name", "type": "A", "template_id": "PING",
    "condition": "zh2zh", "expected": "",
    "prompt": "Reply with the single word: OK",
    "messages": [{"role": "user", "content": "Reply with the single word: OK"}],
}


def main():
    loaded = load_local_env()
    print(f".env loaded keys: {loaded or '(none — is .env present?)'}")
    models = sys.argv[1:] or CANDIDATES
    print(f"Base URL: {OpenAICompatBackend.DEFAULT_BASE} "
          f"(override via PIIAUDITOR_BASE_URL)\n")
    ok = []
    for mid in models:
        try:
            be = OpenAICompatBackend(mid, max_tokens=8, retries=1)
            out = be.complete(PROBE)
            print(f"  [OK]   {mid:26s} -> {out.strip()[:40]!r}")
            ok.append(mid)
        except Exception as e:                       # noqa: BLE001
            msg = str(e).splitlines()[0][:120]
            print(f"  [FAIL] {mid:26s} -> {msg}")
    print()
    if ok:
        print("Usable model IDs:", " ".join(ok))
        print("\nRun a small real pilot, e.g.:")
        print(f"  python cli.py audit --backend dashscope --models {' '.join(ok[:3])} "
              f"--n-entries 20 --max-queries-per-model 400")
    else:
        print("No model responded. Check: (1) .env has the correct key, "
              "(2) the Base URL matches the QwenCloud 'OpenAI Compatible' URL, "
              "(3) your account has access / credit for these models.")


if __name__ == "__main__":
    main()
