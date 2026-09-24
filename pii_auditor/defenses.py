"""Privacy defenses evaluated against the induced-memorization baseline.

Three defense families are implemented, spanning the deployment pipeline:

  D1  Output filtering      (inference-time, no retraining)
  D2  Machine unlearning    (post-training, gradient ascent on the forget set)
  D3  DP-SGD / DP-LoRA      (training-time, Abadi et al. 2016)

Each returns a model (or an output transform) that the standard M3-M6 audit
pipeline can then score, so defended and undefended conditions are measured
with the identical instrument.
"""
from __future__ import annotations

import math
import re
from pathlib import Path
from typing import Callable, Dict, List, Tuple

from .checksums import national_id_is_valid, luhn_is_valid, uscc_is_valid

# --------------------------------------------------------------------------- #
# D1 -- Output filtering (inference-time redaction)
# --------------------------------------------------------------------------- #
# Patterns for the seven CN-PIIBench-Lite categories. The filter is deliberately
# built from the same national-standard format rules an operator would use, so
# the evaluation reflects a realistic deployment-grade scrubber rather than an
# oracle that already knows the ground truth.
_PATTERNS = [
    ("national_id",        re.compile(r"(?<![0-9A-Za-z])[1-9]\d{5}(?:19|20)\d{2}(?:0[1-9]|1[0-2])(?:0[1-9]|[12]\d|3[01])\d{3}[0-9Xx](?![0-9A-Za-z])")),
    ("unionpay_card",      re.compile(r"(?<![0-9])62\d{14,17}(?![0-9])")),
    ("mobile_phone",       re.compile(r"(?<![0-9])1[3-9]\d{9}(?![0-9])")),
    ("social_credit_code", re.compile(r"(?<![0-9A-Za-z])[0-9A-HJ-NPQRTUWXY]{18}(?![0-9A-Za-z])")),
    ("medical_record_id",  re.compile(r"(?<![0-9A-Za-z])[A-Z]{3}\d{14}(?![0-9A-Za-z])")),
    ("soe_employee_id",    re.compile(r"(?<![0-9A-Za-z])[A-Z]{3}\d{9}(?![0-9A-Za-z])")),
]
REDACTION = "[REDACTED]"


def filter_output(text: str, strict: bool = True) -> Tuple[str, int]:
    """Redact Chinese PII from a model completion.

    strict=True additionally validates checksums, redacting only well-formed
    identifiers (fewer false positives, but a leaked value whose checksum the
    model corrupted would survive). strict=False redacts on format alone.
    Returns (filtered_text, n_redactions).
    """
    if not text:
        return text, 0
    n = 0
    out = text
    for cat, pat in _PATTERNS:
        def _sub(m):
            nonlocal n
            val = m.group(0)
            if strict:
                if cat == "national_id" and not national_id_is_valid(val):
                    return val
                if cat == "unionpay_card" and not luhn_is_valid(val):
                    return val
                if cat == "social_credit_code" and not uscc_is_valid(val):
                    return val
            n += 1
            return REDACTION
        out = pat.sub(_sub, out)
    return out, n


def filter_with_normalisation(text: str, strict: bool = True) -> Tuple[str, int]:
    """Filter that also catches separator-obfuscated identifiers.

    A naive scrubber is defeated by a model that emits "1101 0519 ..."; this
    variant folds digit-internal separators before matching, and is used to
    quantify how much leakage a naive filter misses.
    """
    folded = re.sub(r"(?<=[0-9])[\s\-–—](?=[0-9Xx])", "", text or "")
    return filter_output(folded, strict=strict)


def make_output_filter(strict: bool = True, normalise: bool = False) -> Callable[[str], str]:
    fn = filter_with_normalisation if normalise else filter_output
    return lambda t: fn(t, strict=strict)[0]


def over_redaction_rate(benign_texts: List[str], strict: bool = True) -> float:
    """Fraction of benign (PII-free) texts the filter wrongly redacts -- the
    utility cost of the filter."""
    if not benign_texts:
        return 0.0
    hit = sum(1 for t in benign_texts if filter_output(t, strict)[1] > 0)
    return hit / len(benign_texts)


# --------------------------------------------------------------------------- #
# D2 -- Machine unlearning (gradient ascent on the forget set)
# --------------------------------------------------------------------------- #
def unlearn_gradient_ascent(model, tok, forget_texts: List[str],
                            retain_texts: List[str] | None = None,
                            steps: int = 60, lr: float = 1e-4,
                            batch_size: int = 4, max_len: int = 320,
                            retain_weight: float = 1.0, device=None):
    """Gradient-ascent unlearning with optional retain-set descent.

    Follows the most hyperparameter-robust configuration reported by Yao et al.
    (2024): ascend the loss on the forget set (to remove memorized content)
    while descending on a retain set (to preserve general capability).
    Operates on the LoRA parameters, so the base weights are untouched.
    """
    import torch
    model.train()
    # A PeftModel loaded for inference has its adapter frozen
    # (is_trainable=False), leaving no parameters to optimize. Re-enable
    # gradients on the LoRA tensors so the ascent updates the adapter only;
    # the quantized base weights stay frozen.
    n_en = 0
    for name, p in model.named_parameters():
        if "lora" in name.lower():
            p.requires_grad_(True)
            if p.dtype in (torch.float16, torch.bfloat16):
                p.data = p.data.float()      # fp32 for a stable ascent
            n_en += 1
    params = [p for p in model.parameters() if p.requires_grad]
    if not params:
        raise RuntimeError(
            "no trainable parameters found for unlearning; expected LoRA tensors")
    print(f"    [unlearn] {n_en} LoRA tensors enabled for gradient ascent")
    opt = torch.optim.AdamW(params, lr=lr)
    dev = device or next(model.parameters()).device
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()
    prev_cache = getattr(model.config, "use_cache", None)
    model.config.use_cache = False

    def encode(batch):
        enc = tok(batch, return_tensors="pt", padding=True, truncation=True,
                  max_length=max_len).to(dev)
        # padded positions must not contribute to the loss
        labels = enc["input_ids"].clone()
        labels[enc["attention_mask"] == 0] = -100
        return enc, labels

    def batches(texts, bs):
        return [texts[i:i + bs] for i in range(0, len(texts), bs)]

    f_iter = batches(forget_texts, batch_size)
    r_iter = batches(retain_texts or [], batch_size)
    history = []
    for step in range(steps):
        enc, labels = encode(f_iter[step % len(f_iter)])
        out = model(**enc, labels=labels)
        loss = -out.loss                      # ASCEND on the forget set
        if r_iter:
            renc, rlabels = encode(r_iter[step % len(r_iter)])
            rout = model(**renc, labels=rlabels)
            loss = loss + retain_weight * rout.loss   # DESCEND on the retain set
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        history.append({"step": step, "forget_loss": float(out.loss.item())})
        if step % 20 == 0:
            print(f"    [unlearn] step {step}/{steps} forget_loss={out.loss.item():.3f}")
    if prev_cache is not None:
        model.config.use_cache = prev_cache
    model.eval()
    return model, history


# --------------------------------------------------------------------------- #
# D3 -- DP-SGD / DP-LoRA (Abadi et al., 2016)
# --------------------------------------------------------------------------- #
def dp_sgd_finetune(model, tok, texts: List[str], epochs: int = 30,
                    lr: float = 3e-4, clip_norm: float = 1.0,
                    noise_multiplier: float = 1.0, batch_size: int = 8,
                    max_len: int = 320, device=None, ckpt_dir=None):
    """Differentially private fine-tuning of the LoRA parameters.

    Implements DP-SGD directly (Abadi et al., 2016): per-example gradients are
    clipped to an L2 norm of `clip_norm`, summed over the lot, perturbed with
    Gaussian noise of scale `noise_multiplier * clip_norm`, and averaged before
    the optimizer step. Per-example gradients are obtained with microbatches of
    size one, which is slower than vectorized clipping but transparent and
    dependency-free.

    With `ckpt_dir` set, the adapter weights, the optimizer state, the epoch
    counter and both RNG streams are written there after every epoch, and a
    later call with the same directory resumes from the last completed epoch.
    Per-example gradients make this stage cost hours, and a checkpoint written
    only at the end means a dropped connection at epoch 29 of 30 costs all of
    it. The privacy accounting is untouched: epsilon is a function of the noise
    multiplier, the sampling rate and the number of steps composed, none of
    which a restart changes - the same 30 epochs of steps are taken either way.
    Restoring the RNG streams makes a resumed run follow the same trajectory as
    an uninterrupted one, up to nondeterministic CUDA kernels.
    """
    import torch
    if ckpt_dir:
        # only the LoRA tensors are saved, so a checkpoint is megabytes rather
        # than the gigabytes a full state dict would cost on every epoch
        from peft import get_peft_model_state_dict, set_peft_model_state_dict
    model.train()
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=lr)
    dev = device or next(model.parameters()).device
    n = len(texts)
    steps_per_epoch = max(1, n // batch_size)
    history = []
    start_ep = 0
    total_micro = epochs * steps_per_epoch * batch_size

    ck = Path(ckpt_dir) / "dp_state.pt" if ckpt_dir else None
    if ck and ck.exists():
        # onto the CPU, not onto dev: an RNG state is a ByteTensor and both
        # torch.set_rng_state and torch.cuda.set_rng_state_all reject one that
        # has been moved to the GPU. The weights and the optimizer moments do
        # not need placing by hand - load_state_dict copies into the parameters
        # that already live on the right device
        st = torch.load(ck, map_location="cpu", weights_only=False)
        assert st["noise_multiplier"] == noise_multiplier and \
            st["clip_norm"] == clip_norm and st["batch_size"] == batch_size, (
                f"checkpoint at {ck} was written under a different mechanism "
                f"(noise={st['noise_multiplier']}, clip={st['clip_norm']}, "
                f"batch={st['batch_size']}) - its epsilon is not the one being "
                f"claimed here; delete it or fix the arguments")
        set_peft_model_state_dict(model, st["adapter"])
        opt.load_state_dict(st["optimizer"])
        torch.set_rng_state(st["cpu_rng"].cpu().to(torch.uint8))
        if st.get("cuda_rng") is not None and torch.cuda.is_available():
            torch.cuda.set_rng_state_all(
                [s.cpu().to(torch.uint8) for s in st["cuda_rng"]])
        history = st["history"]
        start_ep = st["epoch"] + 1
        print(f"    [dp-sgd] resuming from {ck} at epoch {start_ep + 1}/{epochs} "
              f"({start_ep * steps_per_epoch * batch_size} per-example gradients "
              f"already spent, not repeated)")

    print(f"    [dp-sgd] {epochs} epochs x {steps_per_epoch} steps x {batch_size} "
          f"microbatches = {total_micro} per-example gradient computations "
          f"(this stage is inherently slow: per-example clipping requires one "
          f"backward pass per example)")
    if ck:
        print(f"    [dp-sgd] checkpointing to {ck} after every epoch")
    else:
        print(f"    [dp-sgd] NO checkpoint directory: an interruption loses "
              f"the whole run")
    import time as _t
    _t0 = _t.time()

    for ep in range(start_ep, epochs):
        perm = torch.randperm(n).tolist()
        for s in range(steps_per_epoch):
            lot = [texts[i] for i in perm[s * batch_size:(s + 1) * batch_size]]
            if not lot:
                continue
            accum = [torch.zeros_like(p) for p in params]
            lot_loss = 0.0
            for ex in lot:                       # microbatch of size 1
                enc = tok([ex], return_tensors="pt", truncation=True,
                          max_length=max_len).to(dev)
                out = model(**enc, labels=enc["input_ids"])
                opt.zero_grad(); out.loss.backward()
                # per-example clipping
                sq = sum((p.grad.detach() ** 2).sum() for p in params if p.grad is not None)
                total = torch.sqrt(sq) + 1e-12
                scale = min(1.0, clip_norm / float(total))
                for a, p in zip(accum, params):
                    if p.grad is not None:
                        a += p.grad.detach() * scale
                lot_loss += float(out.loss.item())
            # Gaussian mechanism + average over the lot
            opt.zero_grad()
            sigma = noise_multiplier * clip_norm
            for a, p in zip(accum, params):
                if sigma > 0:
                    noise = torch.normal(0.0, sigma, size=a.shape,
                                         device=a.device, dtype=a.dtype)
                    p.grad = (a + noise) / len(lot)
                else:
                    # noise_multiplier 0 is the mechanistic control of Stage 3
                    # (clipping without noise). Skipping the draw avoids relying
                    # on torch.normal accepting a zero standard deviation, and
                    # is exactly equivalent to adding a zero tensor.
                    p.grad = a / len(lot)
            opt.step()
            history.append({"epoch": ep, "step": s, "loss": lot_loss / len(lot)})
        # rate has to come from this session's own work, not from the epochs a
        # previous session paid for, or a resumed run reports a fictitious eta
        this_run = (ep + 1 - start_ep) * steps_per_epoch * batch_size
        done = (ep + 1) * steps_per_epoch * batch_size
        el = _t.time() - _t0
        eta = el / max(1, this_run) * (total_micro - done) / 60
        print(f"    [dp-sgd] epoch {ep+1}/{epochs} loss={lot_loss/max(1,len(lot)):.3f} "
              f"elapsed={el/60:.1f}min eta={eta:.0f}min")
        if ck:
            ck.parent.mkdir(parents=True, exist_ok=True)
            tmp = ck.with_suffix(".tmp")
            torch.save({"epoch": ep,
                        "adapter": get_peft_model_state_dict(model),
                        "optimizer": opt.state_dict(),
                        "cpu_rng": torch.get_rng_state(),
                        "cuda_rng": (torch.cuda.get_rng_state_all()
                                     if torch.cuda.is_available() else None),
                        "history": history,
                        "noise_multiplier": noise_multiplier,
                        "clip_norm": clip_norm,
                        "batch_size": batch_size,
                        "epochs": epochs}, tmp)
            tmp.replace(ck)          # rename, so a crash mid-write keeps the last good one
            print(f"    [dp-sgd] checkpoint saved at epoch {ep+1}")
    model.eval()
    return model, history


def rdp_epsilon(steps: int, sampling_rate: float, noise_multiplier: float,
                delta: float = 1e-5) -> float:
    """Approximate (eps, delta) via the RDP accountant of the subsampled
    Gaussian mechanism (Mironov et al., 2019), scanned over integer orders.

    Reported as an indicative privacy budget for the configurations evaluated;
    it is the standard accounting used alongside DP-SGD.
    """
    if noise_multiplier <= 0 or sampling_rate <= 0:
        return float("inf")
    q, sigma = sampling_rate, noise_multiplier
    best = float("inf")
    for alpha in range(2, 256):
        # RDP of the subsampled Gaussian mechanism (standard upper bound)
        rdp = q * q * alpha / (2 * sigma * sigma)
        eps = steps * rdp + math.log1p(-1.0 / alpha) - (math.log(delta) + math.log(alpha)) / (alpha - 1)
        best = min(best, eps)
    return max(0.0, best)


# --------------------------------------------------------------------------- #
# Utility probe -- capability retention after a defense
# --------------------------------------------------------------------------- #
# IMPORTANT: the retain set (descended on during unlearning) and the utility
# probe (used only for measurement) MUST be disjoint. Using one set for both
# trains the model on its own test probe and drives measured perplexity toward
# 1.0, which reports memorization of the probe rather than capability retention.
# Retain set: descended on during unlearning to preserve general capability.
RETAIN_TEXTS = [
    "黄河流经中国北方多个省份，是中华文明的重要发源地之一。",
    "数据库管理系统负责数据的存储、检索以及并发访问控制。",
    "细胞是生物体结构和功能的基本单位，由细胞膜包裹而成。",
    "上海是中国重要的金融中心和国际航运枢纽城市。",
    "操作系统负责管理计算机的硬件资源并为程序提供运行环境。",
    "牛顿第二定律指出物体的加速度与所受合力成正比。",
    "统计学通过样本推断总体的特征以及不确定性的大小。",
    "唐诗是中国古典文学的重要组成部分，风格多样内容丰富。",
]

# Held-out capability probe: used ONLY for measurement, never for training or
# retention. Kept unchanged from the original evaluation so that perplexity
# figures remain comparable across all defense conditions.
UTILITY_TEXTS = [
    "人工智能是研究如何使计算机模拟人类智能的科学技术领域。",
    "北京是中华人民共和国的首都，也是全国的政治和文化中心。",
    "机器学习模型通过在大量数据上训练来学习数据中的统计规律。",
    "水的化学式是H2O，由两个氢原子和一个氧原子组成。",
    "长江是中国最长的河流，全长约六千三百公里。",
    "计算机网络通过协议实现不同设备之间的数据交换与通信。",
    "光合作用是植物利用光能将二氧化碳和水转化为有机物的过程。",
    "经济学研究资源的稀缺性以及社会如何分配这些资源。",
]


def perplexity(model, tok, texts: List[str] = None, max_len: int = 256) -> float:
    """Mean perplexity on held-out generic Chinese text: the capability
    reference used to quantify the utility cost of each defense."""
    import torch
    texts = texts or UTILITY_TEXTS
    model.eval()
    dev = next(model.parameters()).device
    tot, cnt = 0.0, 0
    with torch.no_grad():
        for t in texts:
            enc = tok([t], return_tensors="pt", truncation=True,
                      max_length=max_len).to(dev)
            out = model(**enc, labels=enc["input_ids"])
            tot += float(out.loss.item()); cnt += 1
    return math.exp(tot / max(1, cnt))


if __name__ == "__main__":
    # Filter self-test on synthetic values.
    from .m1_generator import build_dataset
    ds = build_dataset(3)
    p = ds["persons"][0]
    txt = (f"该员工身份证号码是{p['national_id']}，手机号码{p['mobile_phone']}，"
           f"工资卡号{p['unionpay_card']}。")
    filt, n = filter_output(txt)
    print("original:", txt)
    print("filtered:", filt, f"({n} redactions)")
    assert p["national_id"] not in filt and p["mobile_phone"] not in filt
    # separator obfuscation defeats the naive filter but not the normalising one
    obf = " ".join(p["national_id"])
    assert filter_output(obf)[1] == 0, "naive filter should miss spaced digits"
    assert filter_with_normalisation(obf)[1] == 1, "normalising filter should catch it"
    print("benign over-redaction:", over_redaction_rate(UTILITY_TEXTS))
    print("rdp epsilon (1000 steps, q=0.1, sigma=1.0):",
          round(rdp_epsilon(1000, 0.1, 1.0), 2))
    print("defenses.py self-test passed")
