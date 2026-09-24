"""M6 -- Risk Classifier and Report Generator.

Consumes the metric tables from M5 and emits the audit deliverables:

    summary.csv        one row per model: aggregate RW-MER, mean CLMD, risk band
    per_category.csv   MER(zh/en), CLMD, RW-MER per (model, category)
    mer_long.csv       tidy MER(model, category, condition)
    results.json       machine-readable bundle consumed by the dashboard
    audit_report.html  human-readable standalone audit report

When the run used the simulation backend, every deliverable is stamped
"SIMULATED / DEMONSTRATION -- NOT REAL MODEL OUTPUT" so results are never
mistaken for real measurements.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

import pandas as pd

from .pri import PRI, CATEGORY_LABELS, CATEGORY_ORDER
from .m5_metrics import RISK_THRESHOLDS

BANNER_SIM = "SIMULATED / DEMONSTRATION — NOT REAL MODEL OUTPUT"


def _band_color(band: str) -> str:
    return {"Low": "#1a7f37", "Medium": "#bf8700", "High": "#cf222e"}.get(band, "#666")


def write_reports(metrics: Dict, meta: Dict, sample_io: List[Dict],
                  out_dir: str | Path) -> Dict[str, str]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = metrics["summary"]
    per_cat = metrics["per_category"]
    mer = metrics["mer"]

    # ---- CSV exports ----------------------------------------------------- #
    summary.to_csv(out_dir / "summary.csv", index=False, encoding="utf-8-sig")
    per_cat.to_csv(out_dir / "per_category.csv", index=False, encoding="utf-8-sig")
    mer.to_csv(out_dir / "mer_long.csv", index=False, encoding="utf-8-sig")

    # ---- results.json (dashboard) --------------------------------------- #
    results = {
        "meta": meta,
        "banner": BANNER_SIM if meta.get("is_simulated") else None,
        "thresholds": RISK_THRESHOLDS,
        "pri": PRI,
        "category_order": CATEGORY_ORDER,
        "category_labels": {c: {"en": CATEGORY_LABELS[c][0],
                                "zh": CATEGORY_LABELS[c][1]} for c in CATEGORY_ORDER},
        "summary": summary.to_dict(orient="records"),
        "per_category": per_cat.where(pd.notna(per_cat), None).to_dict(orient="records"),
        "sample_io": sample_io,
    }
    (out_dir / "results.json").write_text(
        json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")

    # ---- HTML audit report ---------------------------------------------- #
    html = _render_html(results)
    (out_dir / "audit_report.html").write_text(html, encoding="utf-8")

    return {
        "summary_csv": str(out_dir / "summary.csv"),
        "per_category_csv": str(out_dir / "per_category.csv"),
        "mer_csv": str(out_dir / "mer_long.csv"),
        "results_json": str(out_dir / "results.json"),
        "audit_report_html": str(out_dir / "audit_report.html"),
    }


def _render_html(results: Dict) -> str:
    meta = results["meta"]
    banner = results["banner"]
    labels = results["category_labels"]
    rows_summary = ""
    for r in results["summary"]:
        color = _band_color(r["risk_band"])
        rows_summary += (
            f"<tr><td>{r['model']}</td>"
            f"<td style='text-align:right'>{r['aggregate_rwmer']:.4f}</td>"
            f"<td style='text-align:right'>{r['mean_clmd']:+.4f}</td>"
            f"<td><span style='background:{color};color:#fff;padding:2px 10px;"
            f"border-radius:10px;font-weight:600'>{r['risk_band']}</span></td></tr>")

    rows_cat = ""
    for r in results["per_category"]:
        def fmt(x):
            return "—" if x is None else f"{x:.3f}"
        lab = labels.get(r["category"], {}).get("en", r["category"])
        rows_cat += (
            f"<tr><td>{r['model']}</td><td>{lab}</td>"
            f"<td style='text-align:right'>{r['pri']:.3f}</td>"
            f"<td style='text-align:right'>{fmt(r['mer_zh2zh'])}</td>"
            f"<td style='text-align:right'>{fmt(r['mer_en2zh'])}</td>"
            f"<td style='text-align:right'>{fmt(r['clmd'])}</td>"
            f"<td style='text-align:right'>{fmt(r['rwmer'])}</td></tr>")

    banner_html = ("" if not banner else
                   f"<div style='background:#cf222e;color:#fff;padding:10px 16px;"
                   f"border-radius:8px;font-weight:700;margin-bottom:16px'>{banner}</div>")

    return f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<title>PII-Auditor-CN-Lite Audit Report</title>
<style>
 body{{font-family:-apple-system,Segoe UI,Roboto,Helvetica,Arial,sans-serif;
   max-width:960px;margin:32px auto;padding:0 20px;color:#1f2328;line-height:1.5}}
 h1{{font-size:22px}} h2{{font-size:17px;margin-top:28px;border-bottom:1px solid #d0d7de;padding-bottom:4px}}
 table{{border-collapse:collapse;width:100%;margin-top:10px;font-size:14px}}
 th,td{{border:1px solid #d0d7de;padding:6px 10px}} th{{background:#f6f8fa;text-align:left}}
 .meta{{color:#57606a;font-size:13px}}
</style></head><body>
{banner_html}
<h1>PII-Auditor-CN-Lite — PIPL Memorization Leakage Audit</h1>
<p class="meta">Backend: <b>{meta.get('backend')}</b> &middot; Models: {', '.join(meta.get('models', []))}
 &middot; Entries/model: {meta.get('n_entries')} &middot; Templates: {meta.get('n_templates')}
 &middot; Seed: {meta.get('seed')} &middot; Decoding: greedy (temperature 0)</p>
<h2>Model Risk Summary (aggregate RW-MER)</h2>
<table><tr><th>Model</th><th>Aggregate RW-MER</th><th>Mean CLMD</th><th>PIPL Risk</th></tr>
{rows_summary}</table>
<p class="meta">Risk bands: Low &lt; {RISK_THRESHOLDS['low']} &le; Medium &lt; {RISK_THRESHOLDS['high']} &le; High.
 CLMD = MER(EN&rarr;ZH) &minus; MER(ZH&rarr;ZH); positive values indicate a cross-lingual safety gap.</p>
<h2>Per-Category Metrics</h2>
<table><tr><th>Model</th><th>Category</th><th>PRI</th><th>MER ZH&rarr;ZH</th>
<th>MER EN&rarr;ZH</th><th>CLMD</th><th>RW-MER</th></tr>
{rows_cat}</table>
<p class="meta">All PII is synthetic and fictitious. The toolkit is an external black-box wrapper;
 model weights are never accessed or modified.</p>
</body></html>"""


if __name__ == "__main__":
    print("m6_report.py: use via pipeline.run()")
