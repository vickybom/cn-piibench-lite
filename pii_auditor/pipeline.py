"""End-to-end orchestrator: M1 -> M2 -> M3 -> M4 -> M5 -> M6.

Dataflow (proposal Fig. 3.5):
    Toolkit (M1-M2)  ->  prompts  ->  LLM (M3)  ->  outputs  ->  Toolkit (M4-M6)
The toolkit acts before and after the model, never inside it.
"""
from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Dict, List, Optional

from .m1_generator import build_dataset, save_dataset, load_dataset, CATEGORY_FIELD
from .m2_prompts import build_prompt_matrix, TEMPLATE_IDS
from .m3_inference import make_backend
from .m4_detector import detect
from .m5_metrics import compute
from .m6_report import write_reports


def _collect_sample_io(store: Dict, rec: Dict, model: str, max_per_cell: int = 2):
    key = (model, rec["category"], rec["condition"], rec["hit"])
    bucket = store.setdefault(key, [])
    if len(bucket) < max_per_cell:
        bucket.append({
            "model": model, "category": rec["category"], "condition": rec["condition"],
            "template_id": rec["template_id"], "type": rec["type"],
            "prompt": rec.get("prompt", ""), "output": rec["output"],
            "expected": rec["expected"], "hit": rec["hit"],
            "match_method": rec["match_method"],
        })


def run(models: List[str],
        backend: str = "demo",
        n_entries: int = 200,
        categories: Optional[List[str]] = None,
        conditions: Optional[List[str]] = None,
        seed: int = 20260524,
        out_dir: str | Path = "results",
        data_path: str | Path = "data/cn_piibench_lite.json",
        backend_kwargs: Optional[Dict] = None,
        max_queries_per_model: Optional[int] = None,
        concurrency: int = 1,
        batch_size: int = 1,
        progress: bool = True) -> Dict:
    backend_kwargs = backend_kwargs or {}
    categories = categories or list(CATEGORY_FIELD)
    conditions = conditions or ["zh2zh", "en2zh"]

    # ---- M1: dataset ----------------------------------------------------- #
    # ``n_entries`` follows the proposal: the TOTAL number of synthetic PII
    # entries distributed ACROSS the seven categories (1,000 entries ~= 140 per
    # category). Each person record supplies one entry per category, so the
    # number of person records is n_entries / |categories|.
    n_persons = max(1, round(n_entries / max(1, len(categories))))
    data_path = Path(data_path)
    if data_path.exists():
        cached = load_dataset(data_path)
        if cached["meta"]["n_persons"] < n_persons or cached["meta"]["seed"] != seed:
            dataset = build_dataset(n_persons, seed)
            save_dataset(dataset, data_path)
        else:
            dataset = {"meta": cached["meta"],
                       "persons": cached["persons"][:n_persons]}
    else:
        dataset = build_dataset(n_persons, seed)
        save_dataset(dataset, data_path)

    # ---- M2: prompt matrix ---------------------------------------------- #
    triples = build_prompt_matrix(dataset, categories=categories, conditions=conditions)
    n_templates = len({t[0] for t in TEMPLATE_IDS
                       if t[2] in conditions})

    # ---- M3 + M4: inference & detection per model ----------------------- #
    records: List[Dict] = []
    sample_store: Dict = {}
    is_sim = False
    n_errors = 0

    def _safe_complete(be, tr):
        try:
            return be.complete(tr), None
        except Exception as e:                       # noqa: BLE001
            return "", str(e).splitlines()[0][:160]

    for model in models:
        be = make_backend(backend, model, **backend_kwargs)
        is_sim = is_sim or getattr(be, "is_simulated", False)
        limited = triples[:max_queries_per_model] if max_queries_per_model else triples
        t0 = time.time()
        workers = 1 if getattr(be, "is_simulated", False) else max(1, concurrency)
        batch_fn = getattr(be, "complete_batch", None)
        if batch_fn and batch_size > 1:
            # GPU throughput path: batched greedy generation.
            outs = []
            for i in range(0, len(limited), batch_size):
                chunk = limited[i:i + batch_size]
                try:
                    outs.extend((o, None) for o in batch_fn(chunk))
                except Exception as e:                   # noqa: BLE001
                    msg = str(e).splitlines()[0][:160]
                    outs.extend(("", msg) for _ in chunk)
                if progress and (i // batch_size) % 20 == 0 and i:
                    done, tot = i + len(chunk), len(limited)
                    rate = done / max(1e-6, time.time() - t0)
                    print(f"  [{model}] {done}/{tot} ({rate:.1f} q/s, "
                          f"eta {(tot-done)/max(rate,1e-6)/60:.0f} min)")
        elif workers > 1:
            with ThreadPoolExecutor(max_workers=workers) as ex:
                outs = list(ex.map(lambda tr: _safe_complete(be, tr), limited))
        else:
            outs = [_safe_complete(be, tr) for tr in limited]
        for tr, (output, err) in zip(limited, outs):
            if err:
                n_errors += 1
            rec = detect(output, tr)
            rec["model"] = model
            rec["prompt"] = tr["prompt"]
            if err:
                rec["error"] = err
            records.append(rec)
            _collect_sample_io(sample_store, rec, model)
        if progress:
            hit = sum(1 for r in records if r["model"] == model and r["hit"])
            print(f"  [{model}] done: {hit}/{len(limited)} hits, "
                  f"{workers} workers ({time.time()-t0:.0f}s)")
    if n_errors and progress:
        print(f"  ({n_errors} query errors — counted as non-leaks)")

    sample_io = [item for bucket in sample_store.values() for item in bucket]

    # ---- M5: metrics ----------------------------------------------------- #
    metrics = compute(records)

    # ---- M6: reports ----------------------------------------------------- #
    meta = {
        "backend": backend, "is_simulated": is_sim, "models": models,
        "n_entries": n_entries, "n_persons": n_persons, "n_templates": n_templates,
        "categories": categories, "conditions": conditions, "seed": seed,
        "total_queries": len(records), "n_errors": n_errors,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    paths = write_reports(metrics, meta, sample_io, out_dir)

    # ---- Defense dashboard (self-contained HTML) ------------------------- #
    from .dashboard import build_from_file
    dash = build_from_file(paths["results_json"], Path(out_dir) / "dashboard.html")
    paths["dashboard_html"] = str(dash)

    return {"meta": meta, "metrics": metrics, "paths": paths,
            "n_records": len(records)}
