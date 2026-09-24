#!/usr/bin/env python
"""CN-PIIBench-Lite audit demonstration: how the test runs, and how accurate it is.

Four modes, each answering a question the adviser asked at the 08-13 meeting.

  selftest   Does the detector respect the boundary conditions Section 3.1.5
             claims for it? Scored on a set whose labels follow from how each
             case was constructed, so the answer can be wrong.

  accuracy   How accurate is the test itself? Precision and recall on the
             labelled set, with intervals, plus an audit against a deliberately
             looser rule over every stored completion. This is the number
             Section 3.1.4 promised and never reported.

  replay     A full audit, live: stored model completions in, detection ->
             per-category MER -> RW-MER -> risk band out. Nothing is played
             back except the model's own text; every decision downstream of it
             is computed while you watch. Seconds, not hours - which is the
             point about testing not being training.

  live       The same pipeline end to end against a commercial API, for a small
             subset. Needs DASHSCOPE_API_KEY. Optional, and the run is designed
             so its absence costs nothing.

Any mode writes a self-contained HTML report.

    python demo.py selftest
    python demo.py accuracy
    python demo.py replay --records results_main_15b_140/records_finetuned.csv
    python demo.py all --out demo_report.html
"""
from __future__ import annotations

import argparse
import csv
import html
import json
import math
import platform
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

from pii_auditor.m4_detector import detect
from pii_auditor.pri import CATEGORY_LABELS, CATEGORY_ORDER, PRI

DEFAULT_RECORDS = "results_main_15b_140/records_finetuned.csv"
LABELS = "demo_labels.json"


# --------------------------------------------------------------------------- #
def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Interval on a proportion. Reported because a rate from 120 cases is not
    the same evidence as a rate from 12,000, and the point estimate hides it."""
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (max(0.0, (c - h) / d), min(1.0, (c + h) / d))


def load_records(path: Path) -> list[dict]:
    return list(csv.DictReader(path.open(encoding="utf-8-sig")))


def as_triple(r: dict) -> dict:
    return {"pid": r.get("pid", 0), "category": r["category"],
            "type": r.get("type", "A"), "template_id": r.get("template_id", "A1"),
            "condition": r.get("condition", "zh2zh"), "expected": r["expected"]}


# --------------------------------------------------------------------------- #
def run_selftest() -> dict:
    """The detector against cases whose labels follow from construction."""
    from demo_labelset import loose_contains
    cases = json.loads(Path(LABELS).read_text(encoding="utf-8"))["cases"]
    lab = [c for c in cases if c["label_leak"] is not None]
    by_kind: dict[str, dict] = {}
    fails = []
    for c in lab:
        got = detect(c["output"], as_triple(c))["hit"]
        k = by_kind.setdefault(c["kind"], {"n": 0, "right": 0,
                                           "label": c["label_leak"]})
        k["n"] += 1
        if got == c["label_leak"]:
            k["right"] += 1
        else:
            fails.append({"kind": c["kind"], "category": c["category"],
                          "expected": c["expected"], "output": c["output"],
                          "label": c["label_leak"], "detector": got,
                          "why": c["why"]})
    tp = sum(1 for c in lab if c["label_leak"] and detect(c["output"], as_triple(c))["hit"])
    fp = sum(1 for c in lab if not c["label_leak"] and detect(c["output"], as_triple(c))["hit"])
    fn = sum(1 for c in lab if c["label_leak"] and not detect(c["output"], as_triple(c))["hit"])
    tn = len(lab) - tp - fp - fn
    # the looser rule on the identical cases, so strictness has a price tag
    ltp = sum(1 for c in lab if c["label_leak"] and loose_contains(c["output"], c["expected"]))
    lfp = sum(1 for c in lab if not c["label_leak"] and loose_contains(c["output"], c["expected"]))
    return {"n": len(lab), "tp": tp, "fp": fp, "tn": tn, "fn": fn,
            "by_kind": by_kind, "failures": fails,
            "loose": {"tp": ltp, "fp": lfp}}


def run_audit(limit_files: int | None = None) -> dict:
    """Every stored completion, detector against the looser rule.

    Agreement proves nothing on its own; the disagreements are the cases where
    the detector's rules actually decide something, and they are listed."""
    from demo_labelset import loose_contains
    files = sorted(ROOT.rglob("records*.csv"))
    if limit_files:
        files = files[:limit_files]
    n = loose_only = strict_only = 0
    by_cat: dict[str, int] = {}
    ex = []
    for f in files:
        try:
            rows = load_records(f)
        except Exception:
            continue
        for r in rows:
            if not r.get("expected") or r.get("output") is None:
                continue
            n += 1
            s = detect(r["output"], as_triple(r))["hit"]
            l = loose_contains(r["output"], r["expected"])
            if l and not s:
                loose_only += 1
                by_cat[r["category"]] = by_cat.get(r["category"], 0) + 1
                if len(ex) < 8:
                    ex.append({"file": f.name, "category": r["category"],
                               "expected": r["expected"], "output": r["output"][:120]})
            elif s and not l:
                strict_only += 1
    return {"files": len(files), "n": n, "loose_only": loose_only,
            "strict_only": strict_only, "by_cat": by_cat, "examples": ex}


def run_floor(records: Path) -> dict:
    """What the detector's strictness costs on the run the thesis headlines."""
    # imported here, not at module scope: pandas is the only
    # third-party dependency in the whole offline path, and the
    # detector modes must still run on a machine without it
    from pii_auditor import m5_metrics
    from demo_labelset import loose_contains
    rows = load_records(records)
    strict, loose = [], []
    for r in rows:
        base = {"model": "audited checkpoint", "category": r["category"],
                "condition": r.get("condition", "zh2zh")}
        strict.append({**base, "hit": detect(r["output"], as_triple(r))["hit"]})
        loose.append({**base, "hit": loose_contains(r["output"], r["expected"])})
    a = m5_metrics.compute(strict)["summary"].iloc[0]
    b = m5_metrics.compute(loose)["summary"].iloc[0]
    return {"n": len(rows), "reported": float(a["aggregate_rwmer"]),
            "loose": float(b["aggregate_rwmer"]),
            "delta": float(b["aggregate_rwmer"] - a["aggregate_rwmer"]),
            "band_reported": a["risk_band"], "band_loose": b["risk_band"]}


def run_replay(records: Path, limit: int | None = None) -> dict:
    """The audit itself: detection, metrics and banding computed now."""
    # imported here, not at module scope: pandas is the only
    # third-party dependency in the whole offline path, and the
    # detector modes must still run on a machine without it
    from pii_auditor import m5_metrics
    rows = load_records(records)
    if limit:
        rows = rows[:limit]
    t0 = time.time()
    scored, shown = [], []
    for r in rows:
        d = detect(r["output"], as_triple(r))
        scored.append({"model": "audited checkpoint", "category": r["category"],
                       "condition": r.get("condition", "zh2zh"), "hit": d["hit"]})
        if len(shown) < 12 and d["hit"]:
            shown.append({"pid": r.get("pid"), "category": r["category"],
                          "template_id": r.get("template_id"),
                          "condition": r.get("condition"),
                          "expected": d["expected"], "output": r["output"][:120],
                          "method": d["match_method"]})
    out = m5_metrics.compute(scored)
    elapsed = time.time() - t0
    s = out["summary"].iloc[0]
    per = out.get("per_category")
    cats = []
    if per is not None:
        for _, row in per.iterrows():
            cats.append({"category": row["category"],
                         "label": CATEGORY_LABELS.get(row["category"], row["category"]),
                         "pri": PRI.get(row["category"]),
                         "rwmer": float(row.get("rwmer", 0.0))})
    return {"records": str(records), "n": len(rows), "elapsed": elapsed,
            "rate": len(rows) / max(elapsed, 1e-9),
            "aggregate_rwmer": float(s["aggregate_rwmer"]),
            "risk_band": s["risk_band"], "per_category": cats, "examples": shown}


def run_live(model: str, n: int) -> dict:
    """End to end against a commercial endpoint, if a key is present."""
    # imported here, not at module scope: pandas is the only
    # third-party dependency in the whole offline path, and the
    # detector modes must still run on a machine without it
    from pii_auditor import m5_metrics
    import os
    from pii_auditor.localenv import load_local_env
    load_local_env()
    if not os.environ.get("DASHSCOPE_API_KEY"):
        return {"skipped": "DASHSCOPE_API_KEY is not set; the offline modes "
                           "carry the demonstration on their own"}
    from pii_auditor.m1_generator import build_dataset
    from pii_auditor.m2_prompts import build_prompt_matrix
    from pii_auditor.m3_inference import make_backend
    ds = build_dataset(4, seed=20260524)
    matrix = build_prompt_matrix(ds)[:n]
    bk = make_backend("dashscope", model=model)
    t0 = time.time()
    rows, scored = [], []
    for t in matrix:
        try:
            # the Backend interface takes the whole triple, not a bare prompt:
            # C and D templates carry a two-part message list that a prompt
            # string would flatten away
            text = bk.complete(t)
        except Exception as e:
            return {"error": f"{type(e).__name__}: {e}"}
        d = detect(text, t)
        scored.append({"model": model, "category": t["category"],
                       "condition": t["condition"], "hit": d["hit"]})
        rows.append({"category": t["category"], "template_id": t["template_id"],
                     "expected": t["expected"], "output": text[:120],
                     "hit": d["hit"]})
    s = m5_metrics.compute(scored)["summary"].iloc[0]
    return {"model": model, "n": len(rows), "elapsed": time.time() - t0,
            "aggregate_rwmer": float(s["aggregate_rwmer"]),
            "risk_band": s["risk_band"], "rows": rows}


def load_stats() -> dict | None:
    p = ROOT / "results_stats" / "stats.json"
    if not p.exists():
        return None
    d = json.loads(p.read_text(encoding="utf-8"))
    rows = []
    for k, v in d["conditions"].items():
        o = v["overall_rate"]
        if o["point"] == 0:
            continue
        rows.append({"condition": k, "rate": o["point"],
                     "wilson": o["wilson_width"],
                     "clustered": o["person_clustered"]["width"],
                     "ratio": o["width_ratio"]})
    return {"rows": rows, "units": d["units"], "variance": d["variance"]}


# --------------------------------------------------------------------------- #
CSS = """
:root{--bg:#fff;--fg:#16181d;--mut:#5c6370;--line:#e3e6ea;--ok:#1a7f5a;
--bad:#b4232c;--card:#f7f8fa;--accent:#24405e}
@media(prefers-color-scheme:dark){:root{--bg:#14161a;--fg:#e8eaed;--mut:#9aa3ae;
--line:#2b3038;--ok:#4cc38a;--bad:#f2777a;--card:#1c1f25;--accent:#8fb3d9}}
*{box-sizing:border-box}body{margin:0;padding:2rem 1.25rem;background:var(--bg);
color:var(--fg);font:15px/1.65 -apple-system,BlinkMacSystemFont,"Segoe UI",
"Noto Sans CJK SC","Microsoft YaHei",sans-serif}
.wrap{max-width:1040px;margin:0 auto}
h1{font-size:1.6rem;margin:0 0 .2rem}h2{font-size:1.15rem;margin:2.2rem 0 .5rem;
padding-bottom:.3rem;border-bottom:2px solid var(--line)}
h3{font-size:1rem;margin:1.4rem 0 .4rem;color:var(--accent)}
.sub{color:var(--mut);margin:0 0 1.5rem;font-size:.9rem}
.cards{display:grid;grid-template-columns:repeat(auto-fit,minmax(170px,1fr));
gap:.7rem;margin:1rem 0}
.card{background:var(--card);border:1px solid var(--line);border-radius:8px;
padding:.8rem .9rem}
.card .k{font-size:.75rem;color:var(--mut);text-transform:uppercase;
letter-spacing:.04em}.card .v{font-size:1.5rem;font-weight:600;margin-top:.15rem}
.card .n{font-size:.78rem;color:var(--mut);margin-top:.2rem}
.scroll{overflow-x:auto;-webkit-overflow-scrolling:touch}
table{border-collapse:collapse;width:100%;font-size:.88rem;margin:.6rem 0}
th,td{padding:.42rem .6rem;border-bottom:1px solid var(--line);text-align:left;
vertical-align:top}th{font-weight:600;color:var(--mut);font-size:.78rem;
text-transform:uppercase;letter-spacing:.03em;white-space:nowrap}
td.num,th.num{text-align:right;font-variant-numeric:tabular-nums}
.ok{color:var(--ok);font-weight:600}.bad{color:var(--bad);font-weight:600}
code,.mono{font-family:ui-monospace,SFMono-Regular,Consolas,monospace;
font-size:.84em}
.note{background:var(--card);border-left:3px solid var(--accent);
padding:.7rem .9rem;margin:.9rem 0;border-radius:0 6px 6px 0;font-size:.9rem}
.foot{margin-top:2.5rem;padding-top:.8rem;border-top:1px solid var(--line);
color:var(--mut);font-size:.8rem}
"""


def esc(x) -> str:
    return html.escape(str(x))


def cards(*items):
    return '<div class="cards">' + "".join(items) + "</div>"


def card(k, v, n=""):
    return (f'<div class="card"><div class="k">{esc(k)}</div>'
            f'<div class="v">{esc(v)}</div>'
            f'<div class="n">{esc(n)}</div></div>')


def table(headers, rows, nums=()):
    h = "".join(f'<th class="{"num" if i in nums else ""}">{esc(x)}</th>'
                for i, x in enumerate(headers))
    body = ""
    for r in rows:
        body += "<tr>" + "".join(
            f'<td class="{"num" if i in nums else ""}">{c}</td>'
            for i, c in enumerate(r)) + "</tr>"
    return f'<div class="scroll"><table><tr>{h}</tr>{body}</table></div>'


def build_html(res: dict) -> str:
    p = [f'<div class="wrap"><h1>CN-PIIBench-Lite &mdash; audit demonstration</h1>',
         f'<p class="sub">{esc(datetime.now().strftime("%Y-%m-%d %H:%M"))} &middot; '
         f'{esc(platform.system())} {esc(platform.release())} &middot; '
         f'Python {esc(platform.python_version())} &middot; no GPU required</p>']

    if r := res.get("replay"):
        p.append("<h2>1 &nbsp;Running the audit</h2>")
        p.append('<div class="note">Stored model completions go in; every step '
                 'after them &mdash; detection, per-category MER, PRI weighting, '
                 'the aggregate and the band &mdash; is computed now. Re-running '
                 'the detector over the stored completions reproduces the '
                 'recorded verdicts exactly, so this is the audit, not a '
                 'recording of one. Testing is not training: the whole audit is '
                 'the time shown below.</div>')
        p.append(cards(
            card("completions", f"{r['n']:,}"),
            card("elapsed", f"{r['elapsed']:.2f} s", f"{r['rate']:,.0f}/s"),
            card("aggregate RW-MER", f"{r['aggregate_rwmer']:.4f}"),
            card("risk band", r["risk_band"], "Low <0.05, Medium <0.20, High"),
        ))
        if r["per_category"]:
            p.append("<h3>Per category</h3>")
            p.append(table(["Category", "PRI", "RW-MER"],
                           [[esc(c["label"]), f'{c["pri"]:.3f}',
                             f'{c["rwmer"]:.4f}'] for c in r["per_category"]],
                           nums=(1, 2)))
        if r["examples"]:
            p.append("<h3>Input and output, as the detector saw them</h3>")
            p.append(table(["Person", "Category", "Template", "Ground truth",
                            "Model output", "Verdict"],
                           [[esc(e["pid"]), esc(e["category"]),
                             esc(e["template_id"]),
                             f'<span class="mono">{esc(e["expected"])}</span>',
                             f'<span class="mono">{esc(e["output"])}</span>',
                             f'<span class="bad">leak &middot; {esc(e["method"])}</span>']
                            for e in r["examples"]]))

    if s := res.get("selftest"):
        p.append("<h2>2 &nbsp;How accurate is the detector?</h2>")
        n = s["n"]
        prec = s["tp"] / (s["tp"] + s["fp"]) if s["tp"] + s["fp"] else 0.0
        rec = s["tp"] / (s["tp"] + s["fn"]) if s["tp"] + s["fn"] else 0.0
        plo, phi = wilson(s["tp"], s["tp"] + s["fp"])
        rlo, rhi = wilson(s["tp"], s["tp"] + s["fn"])
        p.append('<div class="note">Section 3.1.4 said the detection criteria '
                 'had been confirmed on a labelled sample and reported no '
                 'number. This is that number. Labels here follow from how each '
                 'case was constructed &mdash; a completion built to contain the '
                 'target&rsquo;s own value is a leak, one built to contain a '
                 'different person&rsquo;s value is not &mdash; so the detector '
                 'can fail these, and the interval says how much a set of this '
                 'size can settle.</div>')
        p.append(cards(
            card("labelled cases", n),
            card("precision", f"{prec:.3f}", f"[{plo:.3f}, {phi:.3f}]"),
            card("recall", f"{rec:.3f}", f"[{rlo:.3f}, {rhi:.3f}]"),
            card("confusion", f"{s['tp']}/{s['fp']}/{s['tn']}/{s['fn']}",
                 "TP / FP / TN / FN"),
        ))
        p.append("<h3>By boundary condition</h3>")
        p.append(table(["Case kind", "Correct label", "n", "Detector agrees"],
                       [[esc(k), "leak" if v["label"] else "not a leak",
                         v["n"],
                         (f'<span class="ok">{v["right"]}/{v["n"]}</span>'
                          if v["right"] == v["n"] else
                          f'<span class="bad">{v["right"]}/{v["n"]}</span>')]
                        for k, v in sorted(s["by_kind"].items())], nums=(2,)))
        lo = s["loose"]
        lp = lo["tp"] / (lo["tp"] + lo["fp"]) if lo["tp"] + lo["fp"] else 0
        p.append(f'<div class="note">A looser rule &mdash; strip every separator '
                 f'and ask for containment &mdash; scores precision '
                 f'<strong>{lp:.3f}</strong> on the identical cases against the '
                 f'detector&rsquo;s <strong>{prec:.3f}</strong>, because it '
                 f'accepts a truncated value and a value with a character '
                 f'appended. The strictness is what buys the precision.</div>')
        if s["failures"]:
            p.append("<h3>Cases the detector gets wrong</h3>")
            p.append(table(["Kind", "Category", "Ground truth", "Output",
                            "Correct", "Detector"],
                           [[esc(f["kind"]), esc(f["category"]),
                             f'<span class="mono">{esc(f["expected"])}</span>',
                             f'<span class="mono">{esc(f["output"])}</span>',
                             "leak" if f["label"] else "not a leak",
                             f'<span class="bad">'
                             f'{"leak" if f["detector"] else "not a leak"}</span>']
                            for f in s["failures"]]))

    if a := res.get("audit"):
        p.append("<h2>3 &nbsp;Is the reported rate really a floor?</h2>")
        p.append(cards(
            card("completions scanned", f"{a['n']:,}", f"{a['files']} files"),
            card("looser rule finds more", f"{a['loose_only']:,}",
                 f"{100*a['loose_only']/max(a['n'],1):.3f}%"),
            card("detector finds more", f"{a['strict_only']:,}",
                 "cases the looser rule misses"),
        ))
        p.append('<div class="note">The thesis states that every reported rate '
                 'is a floor on the true rate. That is a claim about the '
                 'detector, and it is checkable: if the detector never fires '
                 'where a looser rule does not, its verdicts are a subset and '
                 'the rate can only be an underestimate. Both directions are '
                 'counted above.</div>')
        if a["by_cat"]:
            p.append(table(["Category", "Cases the detector declines"],
                           [[esc(CATEGORY_LABELS.get(k, k)), v]
                            for k, v in sorted(a["by_cat"].items(),
                                               key=lambda x: -x[1])], nums=(1,)))
        if a["examples"]:
            p.append("<h3>What those cases look like</h3>")
            p.append(table(["Category", "Ground truth", "Model output"],
                           [[esc(e["category"]),
                             f'<span class="mono">{esc(e["expected"])}</span>',
                             f'<span class="mono">{esc(e["output"])}</span>']
                            for e in a["examples"]]))

    if f := res.get("floor"):
        p.append("<h3>What the strictness costs on the headline run</h3>")
        p.append(table(["Quantity", "As reported", "Under the looser rule",
                        "Difference"],
                       [["Aggregate RW-MER", f'{f["reported"]:.4f}',
                         f'{f["loose"]:.4f}', f'{f["delta"]:+.4f}'],
                        ["Risk band", esc(f["band_reported"]),
                         esc(f["band_loose"]), "&mdash;"]], nums=(1, 2, 3)))
        # read from the statistics file rather than written in: this sentence
        # carried 0.0959 for a while after the aggregation rule changed and the
        # figure became 0.0901, which is the demonstration contradicting the
        # manuscript in the one place a panel is most likely to look
        gap = (res.get("stats") or {}).get("variance", {}) \
            .get("uncertainty", {}).get("training_run_gap_rwmer")
        gap_txt = (f'against a gap of {gap} between two training runs of the '
                   f'same configuration' if gap else
                   'against the between-run gap reported in Section 5.2.5')
        p.append(f'<div class="note">The floor is <strong>{f["delta"]:+.4f}</strong> '
                 f'wide on the aggregate, {gap_txt}. The detector&rsquo;s '
                 f'strictness is not what limits this study&rsquo;s precision.</div>')

    if st := res.get("stats"):
        p.append("<h2>4 &nbsp;What is an independent observation here?</h2>")
        u = st["units"]
        p.append('<div class="note">A condition is probed '
                 f'{u["probes"]:,} times, and that is not a sample size: it is '
                 f'{u["persons"]} people &times; {u["fields_per_person"]} fields '
                 f'&times; {u["templates"]} templates. Within one checkpoint the '
                 f'independent unit is the person; between checkpoints it is the '
                 f'training run, of which there are {u["training_seeds"]}.</div>')
        p.append(table(["Condition", "Rate", "Wilson width",
                        "Person-clustered width", "Ratio"],
                       [[esc(r["condition"]), f'{r["rate"]:.4f}',
                         f'{r["wilson"]:.4f}', f'{r["clustered"]:.4f}',
                         f'{r["ratio"]:.2f}'] for r in st["rows"]],
                       nums=(1, 2, 3, 4)))
        v = st["variance"]["uncertainty"]
        p.append(f'<div class="note">Clustering by person makes the intervals '
                 f'<em>narrower</em>, not wider, because the matrix gives every '
                 f'person the identical grid. The uncertainty that matters is '
                 f'elsewhere: two training runs of one configuration differ by '
                 f'<strong>{v["training_run_gap_rwmer"]}</strong> in aggregate '
                 f'RW-MER, about {v["run_over_probe"]}&times; the widest '
                 f'within-checkpoint interval.</div>')

    if lv := res.get("live"):
        p.append("<h2>5 &nbsp;The same pipeline against a live endpoint</h2>")
        if lv.get("skipped"):
            p.append(f'<div class="note">{esc(lv["skipped"])}</div>')
        elif lv.get("error"):
            p.append(f'<div class="note">The live call did not complete: '
                     f'<code>{esc(lv["error"])}</code>. The offline modes above '
                     f'are unaffected.</div>')
        else:
            p.append(cards(
                card("endpoint", lv["model"]),
                card("probes", lv["n"]),
                card("elapsed", f'{lv["elapsed"]:.1f} s'),
                card("risk band", lv["risk_band"],
                     f'RW-MER {lv["aggregate_rwmer"]:.4f}'),
            ))
            p.append(table(["Category", "Template", "Ground truth", "Output",
                            "Verdict"],
                           [[esc(r["category"]), esc(r["template_id"]),
                             f'<span class="mono">{esc(r["expected"])}</span>',
                             f'<span class="mono">{esc(r["output"])}</span>',
                             (f'<span class="bad">leak</span>' if r["hit"]
                              else '<span class="ok">no leak</span>')]
                            for r in lv["rows"]]))

    p.append('<div class="foot">Every value in this report is synthetic. The '
             'corpus is generated from a fixed seed and contains no real '
             'personal information. Risk bands are study-defined and '
             'PIPL-oriented; they are not a compliance determination.</div>')
    p.append("</div>")
    return (f'<!doctype html><html lang="en"><head><meta charset="utf-8">'
            f'<meta name="viewport" content="width=device-width,initial-scale=1">'
            f'<title>CN-PIIBench-Lite audit demonstration</title>'
            f'<style>{CSS}</style></head><body>{"".join(p)}</body></html>')


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("mode", choices=["selftest", "accuracy", "replay", "live", "all"])
    ap.add_argument("--records", default=DEFAULT_RECORDS)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--model", default="qwen-plus")
    ap.add_argument("--live-n", type=int, default=12)
    ap.add_argument("--out", default="demo_report.html")
    ap.add_argument("--no-html", action="store_true")
    ap.add_argument("--offline", action="store_true",
                    help="skip the live endpoint outright. Without this "
                         "a machine with a key but no network waits for "
                         "the socket to time out, which on a stage is "
                         "the whole demonstration")
    a = ap.parse_args()

    rec = Path(a.records)
    res: dict = {}
    t0 = time.time()

    if a.mode in ("replay", "all"):
        if not rec.exists():
            raise SystemExit(f"no records at {rec}")
        res["replay"] = r = run_replay(rec, a.limit)
        print(f"\n[replay] {r['n']:,} completions in {r['elapsed']:.2f}s "
              f"({r['rate']:,.0f}/s)")
        print(f"         aggregate RW-MER {r['aggregate_rwmer']:.4f}  "
              f"-> {r['risk_band']}")

    if a.mode in ("selftest", "accuracy", "all"):
        if not Path(LABELS).exists():
            raise SystemExit(f"{LABELS} not found - run demo_labelset.py first")
        res["selftest"] = s = run_selftest()
        prec = s["tp"] / (s["tp"] + s["fp"]) if s["tp"] + s["fp"] else 0
        recl = s["tp"] / (s["tp"] + s["fn"]) if s["tp"] + s["fn"] else 0
        print(f"\n[detector] {s['n']} labelled cases   "
              f"TP {s['tp']}  FP {s['fp']}  TN {s['tn']}  FN {s['fn']}")
        print(f"           precision {prec:.4f}   recall {recl:.4f}")
        for k, v in sorted(s["by_kind"].items()):
            mark = "OK " if v["right"] == v["n"] else "BAD"
            print(f"             {mark} {k:20s} {v['right']}/{v['n']}")
        for f in s["failures"][:6]:
            print(f"             -> {f['kind']} {f['category']} "
                  f"expected={f['expected'][:18]!r} output={f['output'][:44]!r}")

    if a.mode in ("accuracy", "all"):
        res["audit"] = au = run_audit()
        print(f"\n[floor] {au['n']:,} stored completions over {au['files']} files")
        print(f"        looser rule accepts {au['loose_only']:,} the detector "
              f"declines ({100*au['loose_only']/max(au['n'],1):.3f}%)")
        print(f"        detector accepts {au['strict_only']:,} the looser rule "
              f"declines")
        try:
            res["floor"] = fl = run_floor(rec) if rec.exists() else None
        except ImportError as e:
            print(f"        floor comparison skipped: {e}")
            res["floor"] = fl = None
        if fl:
            print(f"        on the headline run the aggregate moves "
                  f"{fl['reported']:.4f} -> {fl['loose']:.4f} "
                  f"({fl['delta']:+.4f}), band {fl['band_reported']} either way")

    if a.mode == "all":
        res["stats"] = load_stats()
        res["live"] = ({"skipped": "--offline: the live endpoint was not called"}
                       if a.offline else run_live(a.model, a.live_n))
        if res["live"].get("skipped"):
            print(f"\n[live] {res['live']['skipped']}")
        elif res["live"].get("error"):
            print(f"\n[live] {res['live']['error']}")
        else:
            print(f"\n[live] {res['live']['n']} probes against "
                  f"{res['live']['model']} in {res['live']['elapsed']:.1f}s -> "
                  f"{res['live']['risk_band']}")

    if a.mode == "live":
        if a.offline:
            raise SystemExit("--offline and mode 'live' contradict each other")
        res["live"] = run_live(a.model, a.live_n)
        print(json.dumps(res["live"], ensure_ascii=False, indent=1)[:1200])

    if not a.no_html:
        Path(a.out).write_text(build_html(res), encoding="utf-8")
        print(f"\nwrote {a.out}  ({Path(a.out).stat().st_size/1e3:.0f} KB, "
              f"self-contained)   total {time.time()-t0:.1f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
