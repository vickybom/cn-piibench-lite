#!/usr/bin/env python
"""Bundle the toolkit into dist/pii_auditor_pkg.zip for upload to a cloud GPU
(Colab / AutoDL). Contains only the code needed to run an audit — no results,
data, or secrets.
"""
from __future__ import annotations

import zipfile
from pathlib import Path

ROOT = Path(__file__).parent
INCLUDE = ["pii_auditor", "cli.py", "run_gpu.py", "finetune_stress.py",
           "ablation_epochs.py", "defense_eval.py", "rerun_unlearning.py", "rerun_main.py", "unlearn_sweep.py", "analyze_records.py", "make_figures.py", "requirements-gpu.txt", "README.md"]
OUT = ROOT / "dist" / "pii_auditor_pkg.zip"


def main():
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(OUT, "w", zipfile.ZIP_DEFLATED) as z:
        for item in INCLUDE:
            p = ROOT / item
            if p.is_dir():
                for f in p.rglob("*"):
                    if f.suffix == ".pyc" or "__pycache__" in f.parts:
                        continue
                    z.write(f, f.relative_to(ROOT))
            elif p.exists():
                z.write(p, p.relative_to(ROOT))
    print(f"wrote {OUT} ({OUT.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
