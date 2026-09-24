#!/usr/bin/env python
"""E22 - what each defense costs on a recognised Chinese benchmark.

WHY THIS EXISTS
---------------
The adviser's comment-3 review, item O: perplexity on generic Chinese text is
not enough to show that a defense preserves useful capability, and a proper
utility evaluation should include at least one recognised Chinese benchmark such
as C-Eval. Item Q is the same objection with teeth: the abstract says DP-SGD
"eliminated leakage entirely at epsilon = 0.75" without saying whether that model
can still do anything. A model that learns nothing cannot leak, and Section 6.5.1
already had to be rewritten once for exactly that reason.

So every defense condition is scored on C-Eval, and privacy is never reported
without the capability number beside it.

WHY C-EVAL AND NOT CMMLU
------------------------
CMMLU is still distributed as a loading script, which datasets 4.0 no longer
executes. C-Eval is served as parquet - one file per subject - so it is read
directly and the dataset library is not in the path at all. Checked against the
Hub before this was written: 52 subjects, 1,346 questions in the val split. The
test split is four times larger but its labels are withheld, so val is the only
usable one.

WHY LIKELIHOOD SCORING AND NOT GENERATION
-----------------------------------------
Asking a base model to emit the letter "B" measures whether it can follow an
instruction as much as whether it knows the answer. Section 5.8.4 of the thesis
is about exactly that confound arising elsewhere in this study. Each option is
therefore scored by the model's own probability at the answer position, which
needs no instruction-following and one forward pass per question.

WHAT IS RECORDED, AND WHY IT MATTERS HERE
-----------------------------------------
Accuracy alone cannot distinguish "knows a quarter of the answers" from "always
says A". Both score 25% on a four-way question. The distribution over predicted
letters travels with every accuracy, and a condition that collapses onto one
letter is flagged rather than reported as a number.

    python capability_eval.py --check-only
    python capability_eval.py --conditions base,undefended
    python capability_eval.py --conditions unlearned,dp_eps075,dp_plateau
"""
from __future__ import annotations

import argparse
import io
import json
import time
import urllib.request
from pathlib import Path

CEVAL_API = "https://huggingface.co/api/datasets/ceval/ceval-exam"
CEVAL_FILE = "https://huggingface.co/datasets/ceval/ceval-exam/resolve/main/{}"
LETTERS = ["A", "B", "C", "D"]

# The operating points Chapter 6 reports, so the capability numbers line up with
# the privacy numbers already in the manuscript.
CONDITIONS = {
    "base":        {"train": None,
                    "note": "released weights, no fine-tuning"},
    "undefended":  {"train": "lora",
                    "note": "the checkpoint every leakage figure is measured on"},
    "unlearned":   {"train": "unlearn", "steps": 60,
                    "note": "60 steps of gradient ascent, Section 6.4"},
    # DP conditions carry their own epoch count. --epochs is the LoRA setting,
    # 30, and letting it reach DP-SGD trains three times the steps the reported
    # condition was trained for - which is a different mechanism with a
    # different epsilon, not the same condition run longer. The 140-record
    # DP-LoRA row in results_final/defense_140.json records steps 2800, and
    # 2800 / (2240 // 8) = 10 epochs; the pilot's meta.dp.epochs agrees.
    "dp_eps075":   {"train": "dp", "noise": 1.0, "clip": 1.0, "epochs": 10,
                    "note": "the abstract's epsilon = 0.75, noise 1.0 over "
                            "2,800 steps"},
    "dp_plateau":  {"train": "dp", "noise": 0.02, "clip": 1.0, "epochs": 10,
                    "note": "noise 0.02 at the same 2,800 steps, so it differs "
                            "from dp_eps075 in the noise alone"},
}
# Output filtering acts on generated text and never touches the weights, so its
# C-Eval score is the undefended model's by construction. Stated, not measured.
SHARES_WEIGHTS = {"filter": "undefended"}


# --------------------------------------------------------------------------- #
# the benchmark
# --------------------------------------------------------------------------- #
def ceval_files() -> list[str]:
    with urllib.request.urlopen(CEVAL_API, timeout=30) as r:
        meta = json.load(r)
    return sorted(s["rfilename"] for s in meta["siblings"]
                  if s["rfilename"].endswith(".parquet"))


def load_ceval(shots: int = 5, cache: Path | None = None):
    """(questions, per-subject few-shot prefixes) from the parquet files.

    No dataset script and no `datasets` dependency: the files are read straight
    from the Hub, which is what makes this work on datasets 4.0.
    """
    import pandas as pd

    files = ceval_files()
    subjects = sorted({f.split("/")[0] for f in files})
    qs, prefix = [], {}
    for i, sub in enumerate(subjects, 1):
        frames = {}
        for split in ("val", "dev"):
            name = f"{sub}/{split}-00000-of-00001.parquet"
            if name not in files:
                continue
            dest = cache / name.replace("/", "__") if cache else None
            if dest is not None and dest.exists():
                raw = dest.read_bytes()
            else:
                with urllib.request.urlopen(CEVAL_FILE.format(name), timeout=120) as r:
                    raw = r.read()
                if dest is not None:
                    dest.parent.mkdir(parents=True, exist_ok=True)
                    dest.write_bytes(raw)
            frames[split] = pd.read_parquet(io.BytesIO(raw))
        if "val" not in frames:
            continue
        if shots and "dev" in frames:
            ex = frames["dev"].head(shots)
            prefix[sub] = "".join(render(r) + r["answer"] + "\n\n"
                                  for _, r in ex.iterrows())
        else:
            prefix[sub] = ""
        for _, r in frames["val"].iterrows():
            qs.append({"subject": sub, "question": r["question"],
                       "A": r["A"], "B": r["B"], "C": r["C"], "D": r["D"],
                       "answer": str(r["answer"]).strip().upper()})
        if i % 10 == 0:
            print(f"      {i}/{len(subjects)} subjects, {len(qs)} questions")
    return qs, prefix


def render(row) -> str:
    return (f"{row['question']}\n"
            f"A. {row['A']}\nB. {row['B']}\nC. {row['C']}\nD. {row['D']}\n答案：")


# --------------------------------------------------------------------------- #
# scoring
# --------------------------------------------------------------------------- #
def score(model, tok, questions, prefix, batch_size: int = 8) -> dict:
    """Argmax over the model's probability of A/B/C/D at the answer position."""
    import torch

    ids = [tok.encode(l, add_special_tokens=False)[0] for l in LETTERS]
    assert len(set(ids)) == 4, (f"the four option letters share token ids {ids}; "
                                f"likelihood scoring cannot separate them")
    # Truncate from the LEFT. The default trims the tail, which here is the
    # "答案：" the whole method scores at - a prompt over the limit would be
    # scored at whatever token happened to survive.
    tok.truncation_side = "left"

    preds, t0 = [], time.time()
    for i in range(0, len(questions), batch_size):
        chunk = questions[i:i + batch_size]
        texts = [prefix.get(q["subject"], "") + render(q) for q in chunk]
        enc = tok(texts, return_tensors="pt", padding=True,
                  truncation=True, max_length=3072).to(model.device)
        with torch.no_grad():
            logits = model(**enc).logits
        # left padding means the last column is the answer position for every row
        last = logits[:, -1, :]
        preds += [LETTERS[int(torch.argmax(last[j, ids]))] for j in range(len(chunk))]
        if i and (i // batch_size) % 20 == 0:
            rate = (i + len(chunk)) / max(1e-6, time.time() - t0)
            print(f"      {i + len(chunk)}/{len(questions)} ({rate:.1f} q/s)")

    correct = sum(p == q["answer"] for p, q in zip(preds, questions))
    from collections import Counter
    dist = Counter(preds)
    modal = dist.most_common(1)[0][1] / len(preds) if preds else 0.0
    by_sub = {}
    for p, q in zip(preds, questions):
        s = by_sub.setdefault(q["subject"], [0, 0])
        s[0] += p == q["answer"]
        s[1] += 1
    return {
        "n": len(questions),
        "accuracy": round(correct / len(questions), 4),
        "chance": 0.25,
        "letter_distribution": {l: dist.get(l, 0) for l in LETTERS},
        "modal_letter_share": round(modal, 4),
        # The answer key is close to uniform - A 310, B 339, C 344, D 353 - so a
        # model that always picks one letter scores between 23% and 26%, which
        # accuracy alone cannot tell apart from chance. Only this flag can.
        "degenerate": modal > 0.60,
        "per_subject": {k: round(v[0] / v[1], 4) for k, v in sorted(by_sub.items())},
    }


# --------------------------------------------------------------------------- #
def environment() -> dict:
    out = {}
    for mod in ("torch", "transformers", "peft", "accelerate", "bitsandbytes",
                "pandas"):
        try:
            out[mod] = getattr(__import__(mod), "__version__", "?")
        except Exception:
            out[mod] = None
    try:
        import torch
        out["gpu"] = (torch.cuda.get_device_name(0)
                      if torch.cuda.is_available() else "cpu")
    except Exception:
        out["gpu"] = "?"
    return out


def check_deps() -> list[str]:
    problems = []
    try:
        import bitsandbytes as bnb
        raw = getattr(bnb, "__version__", "0")
        parts = []
        for piece in raw.split(".")[:3]:
            digits = "".join(c for c in piece if c.isdigit())
            parts.append(int(digits) if digits else 0)
        while len(parts) < 3:
            parts.append(0)
        if tuple(parts) < (0, 46, 1):
            problems.append(f"bitsandbytes {raw} < 0.46.1")
    except Exception as e:
        problems.append(f"bitsandbytes not importable ({type(e).__name__})")
    for mod in ("torch", "transformers", "peft", "pandas", "pyarrow"):
        try:
            __import__(mod)
        except Exception as e:
            problems.append(f"{mod} not importable ({type(e).__name__})")
    return problems


def free():
    try:
        import gc
        import torch
        gc.collect()
        torch.cuda.empty_cache()
    except Exception:
        pass


def corpus_loss(m, tok, texts, max_len: int = 320, batch_size: int = 8) -> float:
    """Mean next-token loss on the fine-tuning corpus.

    This is what separates the two readings of a high benchmark score. A
    defended model can sit near the base model's C-Eval either because the
    defense preserved its capability while it learned the corpus, or because it
    never learned the corpus at all - and a model that learned nothing cannot
    leak, which makes its zero leakage uninformative. C-Eval cannot tell those
    apart. Loss on the training corpus can: a model that fitted the records
    scores far below the base model here, one that did not scores level with it.
    """
    import torch
    m.eval()
    # keep the tail, the way training did. score() switches the tokenizer to
    # left truncation for its own reasons, and if that were still in force here
    # the measurement would be taken on a different slice of each record
    tok.truncation_side = "right"
    total, ntok = 0.0, 0
    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            enc = tok(texts[i:i + batch_size], return_tensors="pt", padding=True,
                      truncation=True, max_length=max_len).to(m.device)
            labels = enc["input_ids"].clone()
            labels[enc["attention_mask"] == 0] = -100      # do not score padding
            # padding is on the left, so the first real token would be predicted
            # from a pad position. Drop it rather than charge the model for it
            first = enc["attention_mask"].argmax(dim=1)
            labels[torch.arange(labels.size(0), device=labels.device), first] = -100
            out = m(**enc, labels=labels)
            # weight each batch by how many tokens it actually predicted, or a
            # batch of short texts counts as much as a batch of long ones
            n = int((labels[:, 1:] != -100).sum())
            total += float(out.loss) * n
            ntok += n
    return total / max(1, ntok)


def backfill_corpus_loss(result, loss_path, model_id, adir, extras, texts, env):
    """Measure corpus loss for conditions that were already scored.

    Writes its own file and never touches capability.json. That matters when
    this runs beside a training session: the trainer rewrites capability.json
    each time a condition finishes, and a second process that read the file
    earlier and writes it back would erase whatever landed in between. Eight
    hours of DP-SGD is not worth a merge conflict, so the two never share a
    file and are combined at the end instead.
    """
    prev = json.loads(loss_path.read_text(encoding="utf-8")) \
        if loss_path.exists() else {"environment": env, "conditions": {}}
    prev["environment"] = env
    done, skipped = list(result["conditions"]), []
    print(f"\n  corpus loss for {len(done)} scored conditions: {done}")
    print(f"  {len(texts)} training texts, no training, adapters reused from disk")
    print(f"  writing to {loss_path} - capability.json is only read\n")

    for name in done:
        if name in prev["conditions"]:
            print(f"    {name:16s} already measured "
                  f"({prev['conditions'][name]['corpus_loss']:.4f})")
            continue
        s = result["conditions"][name]
        if name == "base":
            adapter = None
        elif name in extras:
            adapter = resolve_adapter(extras[name])
        elif s.get("adapter"):
            adapter = resolve_adapter(s["adapter"])
        else:
            adapter = adir / f"adapter_{name}"
            if not (adapter / "adapter_config.json").exists():
                print(f"    {name:16s} SKIPPED - no adapter at {adapter}; pass "
                      f"--extra {name}=<path> if it lives elsewhere")
                skipped.append(name)
                continue
        m, tok = load_model(model_id, adapter)
        val = round(corpus_loss(m, tok, texts), 4)
        del m
        free()
        prev["conditions"][name] = {"corpus_loss": val,
                                    "adapter": str(adapter) if adapter else None}
        loss_path.write_text(json.dumps(prev, ensure_ascii=False, indent=1),
                             encoding="utf-8")
        print(f"    {name:16s} corpus loss {val:.4f}")

    base = prev["conditions"].get("base", {}).get("corpus_loss")
    print(f"\n{'=' * 68}")
    print(f"{'condition':18s}{'C-Eval':>9}{'corpus loss':>13}{'vs base':>10}")
    print("-" * 68)
    for name, s in result["conditions"].items():
        cl = prev["conditions"].get(name, {}).get("corpus_loss")
        if cl is None:
            continue
        gap = f"{cl - base:+.4f}" if base is not None else "-"
        print(f"{name:18s}{s['accuracy']:>9.4f}{cl:>13.4f}{gap:>10}")
    print("\n  level with base here means the adapter never fitted the records, "
          "so its\n  leakage figure says nothing about whether the defense works. "
          "Far below\n  base means it did learn them and the leakage really was "
          "suppressed.")
    if skipped:
        print(f"\n  !! no adapter found for {skipped} - those rows are missing "
              f"from the table\n     above, not equal to base")
    print(f"\nwrote {loss_path}")


def resolve_adapter(path) -> Path:
    """Find the directory that actually holds adapter_config.json.

    Unzipping a folder and uploading it to Drive routinely produces one extra
    level - adapter_dp_pilot/adapter_dp_pilot/adapter_config.json - and the bare
    assertion that used to be here reported only that the file was missing,
    which is true but useless. Descend one level, and if that fails say what is
    in the directory instead.
    """
    p = Path(path)
    if (p / "adapter_config.json").exists():
        return p
    if p.is_dir():
        nested = [d for d in sorted(p.iterdir())
                  if d.is_dir() and (d / "adapter_config.json").exists()]
        if len(nested) == 1:
            print(f"      adapter is one level down, using {nested[0]}")
            return nested[0]
        if len(nested) > 1:
            raise SystemExit(
                f"{p} holds several adapters ({[d.name for d in nested]}); "
                f"point --extra at exactly one of them")

    if not p.exists():
        raise SystemExit(f"{p} does not exist. Check the path, and that the "
                         f"upload to Drive has actually finished.")
    listing = sorted(x.name + ("/" if x.is_dir() else "") for x in p.iterdir())
    raise SystemExit(
        f"no adapter_config.json in {p}, and none one level down.\n"
        f"  what is there: {listing[:12]}\n"
        f"  an adapter directory needs adapter_config.json and "
        f"adapter_model.safetensors side by side. If a .zip is listed above, "
        f"it was uploaded without being extracted.")


def load_model(model_id: str, adapter: Path | None, four_bit: bool = True):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    # left padding set at construction: assigning it afterwards does not always
    # reach the fast tokenizer's backend on transformers 5.x, and here it would
    # put the scored answer position somewhere other than the last column
    tok = AutoTokenizer.from_pretrained(model_id, padding_side="left")
    tok.padding_side = "left"
    assert tok.padding_side == "left", "tokenizer refuses left padding"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    kw = {"device_map": "auto"}
    if four_bit:
        kw["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16)
    m = AutoModelForCausalLM.from_pretrained(model_id, **kw)
    if adapter is not None:
        from peft import PeftModel
        m = PeftModel.from_pretrained(m, str(adapter))
    m.eval()
    return m, tok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-1.5B")
    ap.add_argument("--conditions", default="base,undefended,unlearned,"
                                            "dp_eps075,dp_plateau")
    ap.add_argument("--persons", type=int, default=140)
    ap.add_argument("--seed", type=int, default=20260524)
    ap.add_argument("--train-seed", type=int, default=42)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--repeats", type=int, default=2)
    ap.add_argument("--rank", type=int, default=32)
    ap.add_argument("--shots", type=int, default=5)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--out", default="results_capability")
    ap.add_argument("--adapter-dir", default=None,
                    help="defaults to --out, so a dropped session resumes")
    ap.add_argument("--extra", action="append", default=[], metavar="NAME=PATH",
                    help="score an adapter that already exists instead of "
                         "training one, e.g. --extra dp_pilot=/content/drive/"
                         "MyDrive/piibench/adapter_dp. Repeatable.")
    ap.add_argument("--check-only", action="store_true")
    ap.add_argument("--corpus-loss", action="store_true",
                    help="for every condition already in capability.json, "
                         "measure its loss on the fine-tuning corpus and stop. "
                         "Reuses the adapters on disk, trains nothing.")
    ap.add_argument("--corpus-loss-out", default=None,
                    help="where the corpus-loss file goes; defaults to "
                         "<out>/corpus_loss.json. Point it at your own Drive "
                         "if the results folder is shared read-only.")
    a = ap.parse_args()

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    adir = Path(a.adapter_dir or out)

    print("E22 capability evaluation")
    problems = check_deps()
    if problems:
        print("\n  environment is not ready:")
        for p in problems:
            print(f"    - {p}")
        print('\n  !pip install -q -U "bitsandbytes>=0.46.1"')
        print("  !pip install -q transformers accelerate peft datasets pandas "
              "pyarrow huggingface_hub\n")
        raise SystemExit("stopping before the download rather than after it")

    # corpus loss reads the accuracies out of capability.json and never scores a
    # question, so it has no business fetching the benchmark. Skipping it also
    # means a rerun of that pass cannot fail on a Hub hiccup it does not need.
    qs, prefix = [], {}
    if not a.corpus_loss:
        print("  loading C-Eval (parquet, no dataset script) ...")
        qs, prefix = load_ceval(a.shots, cache=out / "_ceval")
        print(f"  {len(qs)} questions over {len(prefix)} subjects, "
              f"{a.shots}-shot from the dev split")
    if not a.corpus_loss:
        assert len(qs) == 1346, (f"expected 1,346 val questions, got {len(qs)}; "
                                 f"every condition must see the same set")
    if a.check_only:
        print("  check-only: dataset reachable and parsed")
        return

    env = environment()
    print("  environment: " + "  ".join(f"{k} {v}" for k, v in env.items()))

    from pii_auditor.m1_generator import build_dataset
    from pii_auditor.finetune import build_training_texts, lora_finetune
    ds = build_dataset(a.persons, seed=a.seed)
    texts = build_training_texts(ds, repeats=a.repeats)

    res_path = out / "capability.json"
    if res_path.exists():
        result = json.loads(res_path.read_text(encoding="utf-8"))
    elif a.corpus_loss:
        raise SystemExit(f"{res_path} does not exist - corpus loss reads the "
                         f"conditions an earlier run scored, it does not create "
                         f"them")
    else:
        result = {"model": a.model, "environment": env, "shots": a.shots,
                  "n_questions": len(qs), "conditions": {}}
    result["environment"] = env

    todo = [c.strip() for c in a.conditions.split(",") if c.strip()]
    extras = {}
    for e in a.extra:
        k, _, v = e.partition("=")
        assert v, f"--extra needs NAME=PATH, got {e!r}"
        extras[k] = v
        todo.append(k)

    if a.corpus_loss:
        if not result["conditions"]:
            raise SystemExit(
                f"nothing scored yet in {res_path}.\n"
                f"  This reads results that an earlier run wrote. If that run "
                f"was on a different\n  Google account, this account's Drive is "
                f"a different Drive and does not have them:\n"
                f"  share the piibench folder from the account that has the "
                f"results, add a shortcut\n  to My Drive there, and mount again.")
        backfill_corpus_loss(result, Path(a.corpus_loss_out or
                                          (out / "corpus_loss.json")),
                             a.model, adir, extras, texts, env)
        return

    for name in todo:
        if name in extras:
            # An adapter trained in some earlier run. Nothing is retrained; the
            # point is to get a capability number for a checkpoint that already
            # exists rather than spending hours reproducing it.
            spec = {"train": "existing", "path": extras[name],
                    "note": f"pre-existing adapter at {extras[name]}"}
        else:
            spec = CONDITIONS[name]
        if name in result["conditions"]:
            print(f"\n[{name}] already in {res_path}, skipping")
            continue
        print(f"\n[{name}] {spec['note']}")
        adapter = None
        # the mechanism this condition claims, recorded whether it is trained
        # here or reused from disk, so no result can be read without it
        s_extra = {}
        if spec.get("train") == "dp":
            from pii_auditor import defenses as _d
            _st = spec["epochs"] * (len(texts) // 8)
            s_extra = {"dp_epochs": spec["epochs"], "dp_steps": _st,
                       "noise_multiplier": spec["noise"],
                       "clip_norm": spec["clip"], "delta": 1e-5,
                       "epsilon": round(_d.rdp_epsilon(_st, 8 / len(texts),
                                                       spec["noise"]), 4)}

        if spec["train"] == "existing":
            # An adapter trained in some earlier run. Nothing is retrained: the
            # point is a capability number for a checkpoint that already exists,
            # rather than eight hours spent reproducing it.
            adapter = resolve_adapter(spec["path"])
            spec = {"train": None, "note": spec["note"]}

        if spec["train"] == "lora":
            adapter = adir / "adapter_undefended"
            if not (adapter / "adapter_config.json").exists():
                lora_finetune(a.model, texts, adapter, epochs=a.epochs,
                              r=a.rank, seed=a.train_seed)
                free()
            else:
                print("      reusing the adapter already on disk")

        elif spec["train"] in ("unlearn", "dp"):
            from pii_auditor import defenses
            adapter = adir / f"adapter_{name}"
            if (adapter / "adapter_config.json").exists():
                print("      reusing the adapter already on disk")
            elif spec["train"] == "dp":
                m, tok = load_model(a.model, None)
                from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
                m = prepare_model_for_kbit_training(m)
                m.enable_input_require_grads()
                from pii_auditor.finetune import QWEN_TARGETS
                m = get_peft_model(m, LoraConfig(
                    r=a.rank, lora_alpha=2 * a.rank, lora_dropout=0.0,
                    task_type="CAUSAL_LM", target_modules=QWEN_TARGETS))
                print(f"      {s_extra['dp_epochs']} epochs x "
                      f"{len(texts) // 8} steps = {s_extra['dp_steps']} steps, "
                      f"noise {spec['noise']}, clip {spec['clip']}  ->  "
                      f"epsilon {s_extra['epsilon']:.2f} at delta 1e-5")
                # the checkpoint goes next to the adapter, which is on Drive, so
                # it survives the runtime being recycled
                defenses.dp_sgd_finetune(m, tok, texts,
                                         epochs=s_extra["dp_epochs"],
                                         clip_norm=spec["clip"],
                                         noise_multiplier=spec["noise"],
                                         ckpt_dir=adir / f"ckpt_{name}")
                adapter.mkdir(parents=True, exist_ok=True)
                m.save_pretrained(str(adapter))
                tok.save_pretrained(str(adapter))
                del m
                free()
            else:                                    # unlearning
                src = adir / "adapter_undefended"
                assert (src / "adapter_config.json").exists(), \
                    "unlearning starts from the undefended adapter - run that first"
                m, tok = load_model(a.model, src)
                defenses.unlearn_gradient_ascent(m, tok, texts,
                                                 steps=spec["steps"])
                adapter.mkdir(parents=True, exist_ok=True)
                m.save_pretrained(str(adapter))
                tok.save_pretrained(str(adapter))
                del m
                free()

        m, tok = load_model(a.model, adapter)
        s = score(m, tok, qs, prefix, a.batch_size)
        try:
            from pii_auditor.defenses import perplexity
            s["perplexity"] = round(float(perplexity(m, tok)), 2)
        except Exception as e:
            s["perplexity"] = None
            print(f"      perplexity skipped ({type(e).__name__})")
        del m
        free()

        s["note"] = spec["note"]
        # recorded so a later pass can find this exact checkpoint again without
        # guessing at the naming convention
        s["adapter"] = str(adapter) if adapter else None
        s.update(s_extra)
        result["conditions"][name] = s
        res_path.write_text(json.dumps(result, ensure_ascii=False, indent=1),
                            encoding="utf-8")
        flag = "  <-- DEGENERATE" if s["degenerate"] else ""
        print(f"      C-Eval {s['accuracy']:.4f}  (chance {s['chance']})   "
              f"modal letter {s['modal_letter_share']:.1%}{flag}")
        print(f"      letters {s['letter_distribution']}   ppl {s['perplexity']}")

    print(f"\n{'=' * 66}")
    print(f"{'condition':14s}{'C-Eval':>9}{'vs chance':>11}{'modal':>8}{'ppl':>10}")
    print("-" * 66)
    for name, s in result["conditions"].items():
        print(f"{name:14s}{s['accuracy']:>9.4f}{s['accuracy'] - 0.25:>+11.4f}"
              f"{s['modal_letter_share']:>8.1%}{str(s['perplexity']):>10}"
              + ("   DEGENERATE" if s["degenerate"] else ""))
    print(f"\noutput filtering is not listed: it rewrites generated text and "
          f"never touches\nthe weights, so its C-Eval score is the undefended "
          f"model's by construction.")
    print(f"\nwrote {res_path}")


if __name__ == "__main__":
    main()
