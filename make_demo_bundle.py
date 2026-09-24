#!/usr/bin/env python
"""Build the offline package that goes to the defense on a USB stick.

Design rule: every layer can fail and the demonstration still happens.

  1. Python runs           -> demo.py produces a fresh report live
  2. Python does not run   -> demo_report.html is already in the package
  3. The projector is bad  -> the report is one column, no fixed widths
  4. There is no network   -> --offline skips the endpoint outright
  5. There is no GPU       -> nothing in the offline path wants one

The only third-party import on the offline path is pandas, which is checked
before the package is written rather than discovered on the night.

    python make_demo_bundle.py
"""
from __future__ import annotations

import os
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path

ROOT = Path(__file__).parent
OUT = ROOT / "dist" / "piibench_demo.zip"

CODE = ["demo.py", "demo_labelset.py", "demo_labels.json",
        "pii_auditor/__init__.py", "pii_auditor/m1_generator.py",
        "pii_auditor/m2_prompts.py", "pii_auditor/m3_inference.py",
        "pii_auditor/m4_detector.py", "pii_auditor/m5_metrics.py",
        "pii_auditor/checksums.py", "pii_auditor/pri.py",
        "pii_auditor/localenv.py"]
# Every stored completion, not a sample of them. Two files would make the
# package's own audit report 23,520 completions while the report shipped
# beside it says 357,840, and a panel that re-runs the demonstration would see
# the two disagree. 7 MB compressed is a cheap price for that not happening.
# relative to ROOT: rglob yields absolute paths, and writing those as archive
# member names puts the files somewhere the default --records cannot find them
DATA = sorted(p.relative_to(ROOT).as_posix()
              for p in ROOT.rglob("records*.csv"))
DATA += ["results_stats/stats.json"]
PREBUILT = ["demo_report.html"]

README = """# CN-PIIBench-Lite - defense demonstration

Offline package. No GPU, no network, no model download. One dependency: pandas.

## If everything works

    python demo.py all --offline

About thirty seconds. Writes `demo_report.html` next to itself and prints the
same numbers to the console. Open the HTML in any browser.

To include the live commercial endpoint (needs network and DASHSCOPE_API_KEY in
the environment or in a `.env` file beside demo.py):

    python demo.py all

## If something is wrong on the night

| What broke | What to do |
|---|---|
| No network | `python demo.py all --offline` - everything except section 5 |
| No API key | Nothing. The live section reports that it was skipped and the rest runs |
| pandas missing | `demo.py selftest` and `demo.py accuracy` still run - the detector path is pure standard library. `accuracy` prints its precision and its audit and says the one number it had to skip. `replay` does need pandas |
| Python broken | `demo_report.html` in this package is already built. Open it |
| Projector too small | The report is a single column with no fixed widths; wide tables scroll inside their own box |

## The four modes

    python demo.py replay --records results_main_15b_140/records_finetuned.csv
    python demo.py selftest
    python demo.py accuracy
    python demo.py live --model qwen-plus --live-n 12

`replay` audits 11,760 stored completions in under a second: detection, MER,
PRI weighting, aggregate and risk band are all computed then and there. Only
the model's text is stored. Re-running the detector over those completions
reproduces the recorded verdicts exactly on 35,280 records across three files,
which is what makes this an audit rather than a playback.

`selftest` scores the detector on 362 cases whose labels follow from how each
was built. `accuracy` adds the same comparison over all 357,840 stored
completions and reports what the detector's strictness costs on the headline
run. All 62 record files travel with this package, so re-running the
demonstration here reproduces the numbers in the report beside it exactly.

## What is in here

    demo.py                 the demonstration
    demo_labelset.py        rebuilds demo_labels.json
    demo_labels.json        502 cases: 362 labelled, 140 for the disagreement audit
    demo_report.html        a pre-built report, in case Python will not start
    pii_auditor/            generator, prompts, detector, metrics, PRI
    results_main_15b_140/   the stored completions replayed
    results_stats/          the clustered-bootstrap statistics behind section 4

Every value is synthetic. The corpus is generated from a fixed seed and holds
no real personal information. Risk bands are study-defined and PIPL-oriented;
they are not a compliance determination.
"""


# find_module was removed from the import system in Python 3.12, so a finder
# that only defines it is silently inert and every check built on it passes.
# find_spec is the hook that still exists, and assert_blocks below refuses to
# trust this at all until it has seen it deny something.
BLOCKER = '''
import sys
from importlib.abc import MetaPathFinder
DENY = {"torch", "transformers", "peft", "datasets", "accelerate", "bitsandbytes"}
class _Deny(MetaPathFinder):
    def find_spec(self, name, path=None, target=None):
        if name.split(".")[0] in DENY:
            raise ImportError(name + " is blocked: the offline path must not need it")
        return None
sys.meta_path.insert(0, _Deny())
'''


def assert_blocks(tmp: Path) -> None:
    """A denial test that cannot deny proves nothing about what it tested."""
    (tmp / "sitecustomize.py").write_text(BLOCKER, encoding="utf-8")
    env = dict(os.environ, PYTHONPATH=str(tmp), PYTHONIOENCODING="utf-8")
    r = subprocess.run([sys.executable, "-c", "import torch"],
                       capture_output=True, text=True, env=env)
    if r.returncode == 0:
        raise SystemExit("the import blocker does not block; every result "
                         "below it would be meaningless")
    print("    OK  the blocker denies torch, so the checks below can fail")


def check_offline_runs(tmp: Path) -> list[str]:
    """Run each offline mode with the ML stack denied.

    A static scan of import statements cannot separate a module-level import
    from one nested inside a function the offline path never calls, and
    m3_inference has both: it imports torch only when a local backend is
    actually constructed. Running the modes is the only check that means
    anything, and it is the same check the machine at the defense performs.
    """
    (tmp / "sitecustomize.py").write_text(BLOCKER, encoding="utf-8")
    env = dict(os.environ, PYTHONPATH=str(tmp), PYTHONIOENCODING="utf-8")
    bad = []
    for mode in ("selftest", "accuracy", "replay"):
        r = subprocess.run([sys.executable, "demo.py", mode, "--offline",
                            "--no-html"], cwd=ROOT, env=env,
                           capture_output=True, text=True, encoding="utf-8")
        ok = r.returncode == 0
        print(f"    {'OK ' if ok else 'BAD'} {mode:9s} runs with torch, "
              f"transformers and peft denied")
        if not ok:
            bad.append(mode)
            tail = (r.stderr or "").strip().splitlines()
            if tail:
                print(f"        {tail[-1][:110]}")
    return bad


def main():
    missing = [f for f in CODE + DATA if not (ROOT / f).exists()]
    if missing:
        raise SystemExit(f"missing from the repository: {missing}")

    # a stale report in the package is worse than none: build it now
    print("  building a fresh report ...")
    r = subprocess.run([sys.executable, "demo.py", "all", "--offline",
                        "--out", "demo_report.html"],
                       cwd=ROOT, capture_output=True, text=True, encoding="utf-8")
    if r.returncode != 0:
        raise SystemExit(f"demo.py failed, refusing to ship:\n{r.stdout}\n{r.stderr}")

    print("  running the offline modes with the ML stack denied:")
    with tempfile.TemporaryDirectory() as td:
        assert_blocks(Path(td))
        bad = check_offline_runs(Path(td))
    if bad:
        raise SystemExit(f"these modes need the ML stack after all: {bad}")

    OUT.parent.mkdir(exist_ok=True)
    with zipfile.ZipFile(OUT, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("README.md", README)
        for f in CODE + DATA + PREBUILT:
            z.write(ROOT / f, f)
    size = OUT.stat().st_size / 1e6
    print(f"\n  wrote {OUT}  ({size:.1f} MB, {len(CODE + DATA + PREBUILT) + 1} files)")
    print(f"  fits on anything; unzip and run:  python demo.py all --offline")
    return 0


if __name__ == "__main__":
    sys.exit(main())
