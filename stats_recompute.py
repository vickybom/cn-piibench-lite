#!/usr/bin/env python
"""Re-derive the study's uncertainty from the stored per-query records.

WHY THIS EXISTS
---------------
The adviser's comment-3 review rates statistical independence and inference
**Needs major revision**, and two of its points are the same point:

  K. The matched cross-lingual templates probe the *same* records under two
     languages. That is paired binary data, and differences of proportions with
     per-cell Wilson intervals do not account for the pairing.

  L. Pseudoreplication. The study reports thousands of queries and narrow
     intervals, but 11,760 probes are 140 people x 7 fields x 12 templates. A
     query is not an independent observation: twelve templates applied to one
     memorized record are twelve looks at the same thing.

Nothing here needs a GPU or a re-run. Every quantity is recomputed from records
that were kept at the time, which is the whole reason they were kept.

WHAT IT PRODUCES
----------------
  * an explicit statement of the experimental unit, with counts
  * McNemar's exact test on each matched cross-lingual pair
  * person-clustered bootstrap intervals for every headline quantity
  * the design effect: how much narrower the naive interval was
  * a variance decomposition across probe, template, and training seed

THE HONEST ANSWER TO "WHAT IS THE UNIT"
---------------------------------------
It depends on the comparison, and the manuscript has to say so:

  Within one checkpoint  the independent unit is the **person**. Their seven
                         fields were memorized together and are probed twelve
                         ways; the probes are repeated measures on one subject.
                         n = 140, not 11,760.

  Between checkpoints    the independent unit is the **training run**. Section
                         5.6.4 already showed a capacity ordering reversing
                         between two seeds, which is what it looks like when the
                         unit is the run and you have two of them.

    python stats_recompute.py --boot 2000 --out results_stats
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from pii_auditor.pri import PRI

CONDITIONS = {
    "1.5B base (seed 42)":        "results_main_15b_140/records_base.csv",
    "1.5B fine-tuned (seed 42)":  "results_main_15b_140/records_finetuned.csv",
    "1.5B fine-tuned (seed 1337)": "results_replicate_in/records_1.5B_seed1337.csv",
    "3B fine-tuned (seed 1337)":  "results_replicate_in/records_3B_seed1337.csv",
    "Instruct fine-tuned (seed 1337)":
        "results_replicate_in/records_1.5B-Instruct_seed1337.csv",
    "undefended (defense arm)":   "results_defense_140/rec_undefended.csv",
    "DP-LoRA":                    "results_defense_140/rec_dp.csv",
    "unlearned":                  "results_defense_140/rec_unlearned.csv",
}

PAIRS = [("C1", "D1"), ("C2", "D2")]
AB = [f"{t}{i}" for t in "AB" for i in range(1, 5)]
UNANCHORED = ["A2", "A3"]
ANCHORED = [t for t in AB if t not in UNANCHORED]


# --------------------------------------------------------------------------- #
# loading
# --------------------------------------------------------------------------- #
def load(path: Path) -> list[dict]:
    out = []
    with open(path, encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            out.append({"pid": int(r["pid"]), "category": r["category"],
                        "template_id": r["template_id"],
                        "condition": r.get("condition", ""),
                        "hit": r["hit"] in ("True", "true", "1")})
    return out


def index(recs):
    """hit[pid][category][template] -> (hit, condition), plus the person list."""
    d = defaultdict(lambda: defaultdict(dict))
    for r in recs:
        d[r["pid"]][r["category"]][r["template_id"]] = (r["hit"], r["condition"])
    return d, sorted(d)


# --------------------------------------------------------------------------- #
# the statistics, each a function of a person list so the bootstrap can reuse
# --------------------------------------------------------------------------- #
def rate(idx, persons, templates=None, categories=None) -> float:
    hit = tot = 0
    for p in persons:
        for c, tm in idx[p].items():
            if categories and c not in categories:
                continue
            for t, (h, _) in tm.items():
                if templates and t not in templates:
                    continue
                hit += h
                tot += 1
    return hit / tot if tot else float("nan")


def rwmer(idx, persons) -> float:
    """Aggregate RW-MER, matching m5_metrics.compute exactly.

    Per category the worst case over the two probing conditions is weighted by
    that category's PRI, and the aggregate is the unweighted mean of those
    values. Until 2026-08-18 the aggregate weighted them by PRI a second time,
    which squared the Index; that is no longer the definition and this function
    tracks the engine rather than preserving the older behaviour.
    """
    num = den = 0.0
    for c, pri in PRI.items():
        h_zh = n_zh = h_en = n_en = 0
        for p in persons:
            # group on the stored probing condition, which is exactly what
            # m5_metrics.compute groups on
            for _, (h, cond) in idx[p].get(c, {}).items():
                if cond == "zh2zh":
                    h_zh += h
                    n_zh += 1
                else:
                    h_en += h
                    n_en += 1
        vals = [v for v, n in ((h_zh / n_zh if n_zh else None, n_zh),
                               (h_en / n_en if n_en else None, n_en)) if n]
        if not vals:
            continue
        worst = max(v for v in vals if v is not None)
        num += worst * pri
        den += 1
    return num / den if den else float("nan")


def clmd_matched(idx, persons) -> float:
    diffs = [rate(idx, persons, [en]) - rate(idx, persons, [zh]) for en, zh in PAIRS]
    return float(np.mean(diffs))


# --------------------------------------------------------------------------- #
# inference
# --------------------------------------------------------------------------- #
def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (c - h, c + h)


def boot(idx, persons, fn, n_boot: int, seed: int = 20260524) -> dict:
    """Percentile interval from resampling *people*, not probes.

    Resampling probes treats twelve looks at one memorized record as twelve
    independent draws, which is the assumption under challenge. Resampling
    people keeps each subject's probes together, so the interval reflects how
    much the answer would move on a different 140 people.
    """
    rng = np.random.default_rng(seed)
    arr = np.array(persons)
    vals = np.empty(n_boot)
    for i in range(n_boot):
        vals[i] = fn(idx, rng.choice(arr, size=len(arr), replace=True).tolist())
    lo, hi = np.percentile(vals, [2.5, 97.5])
    return {"point": round(float(fn(idx, persons)), 4),
            "lo": round(float(lo), 4), "hi": round(float(hi), 4),
            "width": round(float(hi - lo), 4),
            "boot_sd": round(float(vals.std(ddof=1)), 4)}


def person_spread(idx, persons) -> dict:
    """Why clustering by person moves the interval the way it does.

    If probes were independent Bernoulli draws, a person's own hit rate over
    their 84 probes would vary with sd sqrt(p(1-p)/84). Comparing that to the
    sd actually observed across people says where the dependence lives. A ratio
    near 1 means people vary as independent sampling predicts; well below 1
    means they do not vary much at all, because what drives a hit is the
    template and the category, and the design gives every person the same
    twelve templates over the same seven categories.
    """
    rates, n_each = [], []
    for p in persons:
        h = t = 0
        for tm in idx[p].values():
            for hit, _ in tm.values():
                h += hit
                t += 1
        rates.append(h / t)
        n_each.append(t)
    rates = np.array(rates)
    p_bar = float(rates.mean())
    per_person = int(np.median(n_each))
    expected = float(np.sqrt(p_bar * (1 - p_bar) / per_person)) if per_person else float("nan")
    observed = float(rates.std(ddof=1))
    return {"probes_per_person": per_person,
            "observed_sd_across_people": round(observed, 4),
            "sd_if_probes_were_independent": round(expected, 4),
            "ratio": round(observed / expected, 2) if expected else None}


def mcnemar(idx, persons, en: str, zh: str) -> dict:
    """Exact McNemar on one matched pair.

    Each (person, category) contributes one pair of binary outcomes from the two
    templates that differ only in language. Only discordant pairs carry
    information about the direction, which is exactly what a difference of two
    marginal proportions throws away.
    """
    b = c = both = neither = 0
    for p in persons:
        for cat, tm in idx[p].items():
            if en not in tm or zh not in tm:
                continue
            e, z = tm[en][0], tm[zh][0]     # (hit, condition)
            if e and not z:
                b += 1
            elif z and not e:
                c += 1
            elif e and z:
                both += 1
            else:
                neither += 1
    n_disc = b + c
    if n_disc == 0:
        p_val = 1.0
    else:
        from scipy.stats import binomtest
        p_val = float(binomtest(min(b, c), n_disc, 0.5).pvalue)
    return {"pair": f"{en}-{zh}", "n_pairs": b + c + both + neither,
            "en_only": b, "zh_only": c, "both": both, "neither": neither,
            "discordant": n_disc,
            "diff": round((b - c) / (b + c + both + neither), 4),
            "p_exact": round(p_val, 6)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--boot", type=int, default=2000)
    ap.add_argument("--out", default="results_stats")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    report: dict = {"n_bootstrap": a.boot, "conditions": {}}

    print("=" * 78)
    print("EXPERIMENTAL UNITS")
    print("=" * 78)
    recs = load(Path(CONDITIONS["1.5B fine-tuned (seed 42)"]))
    idx, persons = index(recs)
    cats = {r["category"] for r in recs}
    tmpl = {r["template_id"] for r in recs}
    report["units"] = {"probes": len(recs), "persons": len(persons),
                       "fields_per_person": len(cats), "templates": len(tmpl),
                       "training_seeds": 2}
    print(f"  probes per condition        {len(recs):>7,}")
    print(f"  persons (independent)       {len(persons):>7}")
    print(f"  fields per person           {len(cats):>7}")
    print(f"  templates per field         {len(tmpl):>7}")
    print(f"  training runs per model     {2:>7}")
    print(f"\n  A probe is not an independent observation: {len(recs):,} of them")
    print(f"  are {len(persons)} people looked at {len(cats) * len(tmpl)} ways each.")

    for label, path in CONDITIONS.items():
        p = Path(path)
        if not p.exists():
            print(f"\n  MISSING {path}")
            continue
        recs = load(p)
        idx, persons = index(recs)
        k = sum(r["hit"] for r in recs)
        n = len(recs)

        naive = wilson(k, n)
        clustered = boot(idx, persons, lambda i, ps: rate(i, ps), a.boot)
        w_naive = naive[1] - naive[0]
        deff = clustered["width"] / w_naive if w_naive else float("nan")

        entry = {
            "path": path,
            "overall_rate": {
                "point": round(k / n, 4),
                "wilson_naive": [round(naive[0], 4), round(naive[1], 4)],
                "wilson_width": round(w_naive, 4),
                "person_clustered": clustered,
                "width_ratio": round(deff, 2),
            },
            "person_spread": person_spread(idx, persons),
            "rwmer": boot(idx, persons, rwmer, a.boot),
            "clmd_matched": boot(idx, persons, clmd_matched, a.boot),
            "mcnemar": [mcnemar(idx, persons, en, zh) for en, zh in PAIRS],
            "anchored": round(rate(idx, persons, ANCHORED), 4),
            "unanchored": round(rate(idx, persons, UNANCHORED), 4),
            "per_template": {t: round(rate(idx, persons, [t]), 4) for t in sorted(tmpl)},
        }
        report["conditions"][label] = entry

        print(f"\n{'-' * 78}\n{label}")
        r = entry["overall_rate"]
        print(f"  overall hit rate      {r['point']:.4f}")
        print(f"    Wilson over probes  [{r['wilson_naive'][0]:.4f}, "
              f"{r['wilson_naive'][1]:.4f}]  width {r['wilson_width']:.4f}")
        print(f"    clustered by person [{r['person_clustered']['lo']:.4f}, "
              f"{r['person_clustered']['hi']:.4f}]  width "
              f"{r['person_clustered']['width']:.4f}"
              f"   <- {r['width_ratio']}x the naive width")
        sp = entry["person_spread"]
        print(f"    people vary by sd  {sp['observed_sd_across_people']:.4f} "
              f"against {sp['sd_if_probes_were_independent']:.4f} expected "
              f"if probes were independent  (ratio {sp['ratio']})")
        rw = entry["rwmer"]
        print(f"  aggregate RW-MER      {rw['point']:.4f}  "
              f"[{rw['lo']:.4f}, {rw['hi']:.4f}]")
        cl = entry["clmd_matched"]
        print(f"  matched-pair CLMD     {cl['point']:+.4f} "
              f"[{cl['lo']:+.4f}, {cl['hi']:+.4f}]")
        for m in entry["mcnemar"]:
            print(f"    McNemar {m['pair']}: EN-only {m['en_only']}, "
                  f"ZH-only {m['zh_only']}, discordant {m['discordant']}, "
                  f"p = {m['p_exact']:.4g}")

    # ---- variance across training seeds ----------------------------------- #
    print(f"\n{'=' * 78}\nVARIANCE BY SOURCE\n{'=' * 78}")
    seeds = {}
    for lab in ("1.5B fine-tuned (seed 42)", "1.5B fine-tuned (seed 1337)"):
        if lab in report["conditions"]:
            seeds[lab] = report["conditions"][lab]
    if len(seeds) == 2:
        a42, a13 = (v["rwmer"]["point"] for v in seeds.values())
        probe_w = list(seeds.values())[0]["overall_rate"]["wilson_width"]
        clus_w = list(seeds.values())[0]["overall_rate"]["person_clustered"]["width"]
        tmpl_sd = float(np.std(list(list(seeds.values())[0]["per_template"].values()),
                               ddof=1))
        report["variance"] = {
            "uncertainty": {
                "probe_sampling_wilson_width": round(probe_w, 4),
                "person_clustered_width": round(clus_w, 4),
                "training_run_gap_rwmer": round(abs(a42 - a13), 4),
                "run_over_probe": round(abs(a42 - a13) / probe_w, 1) if probe_w else None,
            },
            "effect_not_noise": {"between_template_sd": round(tmpl_sd, 4)},
        }
        print("  --- uncertainty ---")
        print(f"  probe sampling (Wilson width)      {probe_w:.4f}")
        print(f"  person clustering (boot width)     {clus_w:.4f}")
        print(f"  between training seeds (RW-MER)    {abs(a42 - a13):.4f}"
              f"   <- {abs(a42 - a13) / probe_w:.0f}x the probe interval")
        print("\n  --- structure, not uncertainty ---")
        print(f"  between templates (sd)             {tmpl_sd:.4f}"
              f"   (an effect the design measures, not error)")
        print("\n  The manuscript quoted intervals for the smallest of the three.")

    (out / "stats.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\nwrote {out / 'stats.json'}")


if __name__ == "__main__":
    main()
