#!/usr/bin/env python
"""Rename a condition in capability.json and stamp the mechanism it really ran.

Why this exists. The `dp_eps075` row written on 2026-08-16 was trained for 30
epochs because --epochs, which is the LoRA setting, reached DP-SGD. That is
8,400 steps, not the 2,800 the manuscript's epsilon = 0.75 condition was
trained for, and at 8,400 steps the same accountant returns epsilon = 1.36. The
row is a valid measurement of a different mechanism, so it is relabelled rather
than deleted - and the mechanism is written into the row this time, because a
number stored without its parameters is a number that can be attached to an
epsilon it never had.

    python relabel_condition.py --dir $DRIVE/results_capability \
        --from dp_eps075 --to dp_eps136_8400steps \
        --dp-epochs 30 --noise 1.0 --clip 1.0

Touches capability.json and corpus_loss.json, backs both up first, and never
renames onto a key that already exists.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))


def n_texts(persons: int, seed: int, repeats: int) -> int:
    """The corpus size the run used, needed for the sampling rate."""
    from pii_auditor.m1_generator import build_dataset
    from pii_auditor.finetune import build_training_texts
    return len(build_training_texts(build_dataset(persons, seed=seed),
                                    repeats=repeats))


def rename_in(path: Path, old: str, new: str, extra: dict | None) -> bool:
    if not path.exists():
        print(f"  {path.name}: not there, skipped")
        return False
    d = json.loads(path.read_text(encoding="utf-8"))
    conds = d["conditions"]
    if old not in conds:
        print(f"  {path.name}: no condition named {old!r} "
              f"(has {sorted(conds)}), skipped")
        return False
    assert new not in conds, \
        f"{path.name} already has a condition named {new!r}; refusing to merge"

    bak = path.with_suffix(f".json.bak-{time.strftime('%Y%m%d-%H%M%S')}")
    bak.write_text(json.dumps(d, ensure_ascii=False, indent=1), encoding="utf-8")

    # rebuild in place so the row keeps its position in the file
    conds_new = {}
    for k, v in conds.items():
        if k == old:
            if extra:
                v.update(extra)
            conds_new[new] = v
        else:
            conds_new[k] = v
    d["conditions"] = conds_new
    path.write_text(json.dumps(d, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"  {path.name}: {old} -> {new}   (backup {bak.name})")
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True, help="the results_capability folder")
    ap.add_argument("--from", dest="old", required=True)
    ap.add_argument("--to", dest="new", required=True)
    ap.add_argument("--dp-epochs", type=int, required=True,
                    help="the epoch count the row was ACTUALLY trained for")
    ap.add_argument("--noise", type=float, required=True)
    ap.add_argument("--clip", type=float, default=1.0)
    ap.add_argument("--lot", type=int, default=8)
    ap.add_argument("--persons", type=int, default=140)
    ap.add_argument("--seed", type=int, default=20260524)
    ap.add_argument("--repeats", type=int, default=2)
    a = ap.parse_args()

    from pii_auditor.defenses import rdp_epsilon
    n = n_texts(a.persons, a.seed, a.repeats)
    steps = a.dp_epochs * (n // a.lot)
    eps = rdp_epsilon(steps, a.lot / n, a.noise)
    extra = {"dp_epochs": a.dp_epochs, "dp_steps": steps,
             "noise_multiplier": a.noise, "clip_norm": a.clip,
             "epsilon": round(eps, 4), "delta": 1e-5,
             # the adapter was removed from Drive so the corrected run could not
             # mistake it for its own; the measurements below stand, the weights
             # do not exist any more
             "adapter": None,
             "note": f"{a.dp_epochs} epochs = {steps} steps at noise {a.noise}, "
                     f"epsilon {eps:.2f}. NOT the manuscript's epsilon = 0.75 "
                     f"condition, which is 2,800 steps; kept as the measurement "
                     f"of a more heavily composed mechanism"}

    print(f"\n  corpus {n} texts, lot {a.lot}, {a.dp_epochs} epochs "
          f"= {steps} steps, q = {a.lot / n:.5f}")
    print(f"  noise {a.noise}, clip {a.clip}  ->  epsilon {eps:.4f} at delta 1e-5\n")

    d = Path(a.dir)
    touched = rename_in(d / "capability.json", a.old, a.new, extra)
    rename_in(d / "corpus_loss.json", a.old, a.new, None)

    if touched:
        print("\n  capability.json now holds:")
        c = json.loads((d / "capability.json").read_text(encoding="utf-8"))
        for k, v in c["conditions"].items():
            bits = ""
            if "epsilon" in v:
                bits = (f"   eps {v['epsilon']}  {v['dp_steps']} steps  "
                        f"noise {v['noise_multiplier']}")
            print(f"    {k:22s} C-Eval {v['accuracy']:.4f}{bits}")
        print(f"\n  {a.old!r} is now free, so a corrected run will train it "
              f"rather than skip it")
    return 0


if __name__ == "__main__":
    sys.exit(main())
