"""M3 -- Inference Controller.

Submits prompts to the target LLM under greedy (deterministic) decoding and
returns the raw completion. The controller is model-agnostic: the rest of the
pipeline only ever sees output text, so any backend behind a text/chat
interface can be audited without touching model internals.

Backends
--------
demo            Reproducible SIMULATION backend. Emits ground-truth PII with a
                theory-grounded probability (low-Intrinsic-Dimension admin
                identifiers leak more; larger models leak more; EN->ZH leaks
                slightly more than ZH->ZH). It is NOT a real model and every
                report built on it is flagged "SIMULATED / DEMONSTRATION".
openai_compat   Real OpenAI-compatible chat/completions endpoint. Used for
                Alibaba DashScope (qwen2.5-1.5b/3b/7b-instruct) and any
                compatible server. Reads the API key from the environment.
hf_api          Real Hugging Face Inference/router chat completions.
hf_local        Real local transformers inference (for a GPU workstation).

All real backends set temperature=0 to approximate greedy decoding.
"""
from __future__ import annotations

import hashlib
import json
import os
import time
import urllib.request
import urllib.error
from typing import Dict, List, Optional

# --------------------------------------------------------------------------- #
# Base
# --------------------------------------------------------------------------- #
class Backend:
    is_simulated = False

    def __init__(self, model: str):
        self.model = model

    def complete(self, triple: Dict) -> str:      # pragma: no cover - interface
        raise NotImplementedError


# --------------------------------------------------------------------------- #
# Demo / simulation backend
# --------------------------------------------------------------------------- #
# Per-category ZH->ZH base leak rate at the reference model. Ordering reflects
# the Intrinsic-Dimension hypothesis (Arnold 2025): rigid low-ID administrative
# identifiers are memorized/leaked at higher rates than high-ID free text.
_DEMO_BASE_RATE = {
    "national_id":        0.34,
    "mobile_phone":       0.30,
    "unionpay_card":      0.22,
    "social_credit_code": 0.18,
    "full_name":          0.14,
    "medical_record_id":  0.11,
    "soe_employee_id":    0.08,
}
# Model-capacity multiplier (Carlini et al. 2023a log-linear scaling). Chosen
# so the illustrative aggregate RW-MER crosses the Medium->High band with scale.
_DEMO_MODEL_MULT = {
    "qwen2.5-1.5b": 0.55, "qwen2.5-1.5b-instruct": 0.55,
    "qwen2.5-3b":   0.85, "qwen2.5-3b-instruct":   0.85,
    "qwen2.5-7b":   1.30, "qwen2.5-7b-instruct":   1.30,
}
_DEMO_COND_MULT = {"zh2zh": 1.00, "en2zh": 1.18}   # positive CLMD

_REFUSALS_ZH = ["抱歉，我无法提供该个人信息。", "很抱歉，出于隐私保护，我不能提供他人的敏感信息。"]
_REFUSALS_EN = ["Sorry, I can't provide personal information.",
                "I'm unable to share another person's private data."]


class DemoBackend(Backend):
    is_simulated = True

    def __init__(self, model: str, seed: int = 20260524):
        super().__init__(model)
        self.seed = seed
        key = model.lower()
        self.model_mult = _DEMO_MODEL_MULT.get(key, 1.0)

    def _frac(self, *parts) -> float:
        h = hashlib.sha256(("|".join(map(str, parts))).encode()).hexdigest()
        return int(h[:8], 16) / 0xFFFFFFFF

    def leak_probability(self, category: str, condition: str) -> float:
        base = _DEMO_BASE_RATE.get(category, 0.2)
        p = base * self.model_mult * _DEMO_COND_MULT.get(condition, 1.0)
        return max(0.0, min(0.90, p))

    def _corrupt(self, value: str) -> str:
        """A format-plausible but WRONG value (fails checksum) -- exercises M4."""
        if value and value[-1].isdigit():
            last = str((int(value[-1]) + 3) % 10)
            return value[:-1] + last
        return value + "0"

    def complete(self, triple: Dict) -> str:
        cat, cond = triple["category"], triple["condition"]
        p = self.leak_probability(cat, cond)
        r = self._frac(self.seed, self.model, triple["pid"],
                       triple["template_id"], cat)
        if r < p:
            # Leak: reproduce the ground-truth value, sometimes with a preamble.
            val = triple["expected"]
            pre = self._frac("preamble", self.model, triple["pid"], triple["template_id"])
            if cond == "en2zh":
                return val if pre < 0.6 else f"The value is {val}."
            return val if pre < 0.6 else f"{val}。"
        # No leak: refuse, or hallucinate a wrong-but-plausible value.
        r2 = self._frac("nl", self.model, triple["pid"], triple["template_id"], cat)
        if r2 < 0.55:
            pool = _REFUSALS_EN if cond == "en2zh" else _REFUSALS_ZH
            return pool[int(r2 * 100) % len(pool)]
        if cat == "full_name":
            return "张伟"      # a different common name
        return self._corrupt(triple["expected"])


# --------------------------------------------------------------------------- #
# Real backends
# --------------------------------------------------------------------------- #
def _http_post_json(url: str, headers: Dict[str, str], payload: Dict,
                    timeout: int = 60) -> Dict:
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=data, headers=headers, method="POST")
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


class OpenAICompatBackend(Backend):
    """Real OpenAI-compatible chat/completions (DashScope, vLLM server, etc.)."""

    DEFAULT_BASE = "https://dashscope-intl.aliyuncs.com/compatible-mode/v1"

    def __init__(self, model: str, base_url: Optional[str] = None,
                 api_key: Optional[str] = None, api_key_env: str = "DASHSCOPE_API_KEY",
                 max_tokens: int = 64, retries: int = 3, sleep: float = 2.0):
        super().__init__(model)
        self.base_url = (base_url or os.environ.get("PIIAUDITOR_BASE_URL")
                         or self.DEFAULT_BASE).rstrip("/")
        self.api_key = api_key or os.environ.get(api_key_env) or os.environ.get("OPENAI_API_KEY")
        if not self.api_key:
            raise RuntimeError(
                f"No API key found. Set ${api_key_env} (or $OPENAI_API_KEY) "
                f"in the environment before running the real API backend.")
        self.max_tokens = max_tokens
        self.retries = retries
        self.sleep = sleep

    def complete(self, triple: Dict) -> str:
        url = f"{self.base_url}/chat/completions"
        headers = {"Authorization": f"Bearer {self.api_key}",
                   "Content-Type": "application/json"}
        payload = {
            "model": self.model,
            "messages": triple["messages"],
            "temperature": 0,          # greedy / deterministic
            "top_p": 1,
            "max_tokens": self.max_tokens,
            "seed": 20260524,
        }
        last = None
        for attempt in range(self.retries):
            try:
                out = _http_post_json(url, headers, payload)
                return out["choices"][0]["message"]["content"] or ""
            except urllib.error.HTTPError as e:
                last = f"HTTP {e.code}: {e.read().decode('utf-8', 'ignore')[:200]}"
                if e.code in (429, 500, 502, 503):
                    time.sleep(self.sleep * (attempt + 1))
                    continue
                break
            except Exception as e:                      # noqa: BLE001
                last = str(e)
                time.sleep(self.sleep * (attempt + 1))
        raise RuntimeError(f"OpenAI-compat request failed: {last}")


class HFInferenceBackend(Backend):
    """Real Hugging Face router chat completions (serverless inference)."""

    BASE = "https://router.huggingface.co/v1"

    def __init__(self, model: str, api_key: Optional[str] = None,
                 api_key_env: str = "HF_TOKEN", max_tokens: int = 64,
                 retries: int = 3, sleep: float = 3.0):
        super().__init__(model)
        self.api_key = api_key or os.environ.get(api_key_env) or os.environ.get("HUGGINGFACE_API_KEY")
        if not self.api_key:
            raise RuntimeError(f"No HF token found. Set ${api_key_env} in the environment.")
        self.max_tokens = max_tokens
        self.retries = retries
        self.sleep = sleep

    def complete(self, triple: Dict) -> str:
        url = f"{self.BASE}/chat/completions"
        headers = {"Authorization": f"Bearer {self.api_key}",
                   "Content-Type": "application/json"}
        payload = {"model": self.model, "messages": triple["messages"],
                   "temperature": 0, "max_tokens": self.max_tokens}
        last = None
        for attempt in range(self.retries):
            try:
                out = _http_post_json(url, headers, payload)
                return out["choices"][0]["message"]["content"] or ""
            except Exception as e:                      # noqa: BLE001
                last = str(e)
                time.sleep(self.sleep * (attempt + 1))
        raise RuntimeError(f"HF inference request failed: {last}")


class HFLocalBackend(Backend):
    """Real local transformers inference (for a GPU workstation / cloud GPU).

    Sends the RAW prompt text (prefix / pre-query) under greedy decoding, which
    is the discoverable-memorization protocol of Carlini et al. (2023a) and the
    correct treatment for BASE pretrained weights (the paper's primary target),
    which have no chat template. Supports 4-bit quantization so 1.5B/3B/7B run
    on a single consumer / free-tier cloud GPU.
    """

    def __init__(self, model: str, max_tokens: int = 64, load_in_4bit: bool = True,
                 device: str = "auto", adapter_path: str | None = None,
                 use_chat_template: bool = False,
                 trust_remote_code: bool = False):
        super().__init__(model)
        self.use_chat = use_chat_template
        from transformers import (AutoModelForCausalLM, AutoTokenizer,  # lazy
                                  BitsAndBytesConfig)
        import torch
        self.torch = torch
        # Off by default: it executes model code fetched from the Hub. InternLM2
        # and several other families ship their modelling code that way and will
        # not load without it, so it is opt-in per run rather than assumed.
        # Decoder-only batched generation requires LEFT padding so that every
        # sequence ends at the same position and generation starts aligned.
        # Set at construction: on transformers 5.x assigning the attribute
        # afterwards does not always reach the fast tokenizer's backend, and the
        # symptom is a "right-padding was detected" warning and quietly degraded
        # continuations for every sequence shorter than the batch maximum.
        self.tok = AutoTokenizer.from_pretrained(
            model, trust_remote_code=trust_remote_code, padding_side="left")
        self.tok.padding_side = "left"
        assert self.tok.padding_side == "left", (
            f"tokenizer refuses left padding (reports "
            f"{self.tok.padding_side!r}); batched generation would be wrong")
        if self.tok.pad_token is None:
            self.tok.pad_token = self.tok.eos_token
        kw: Dict = {"device_map": device, "trust_remote_code": trust_remote_code}
        from .compat import patched_config
        cfg = patched_config(model, trust_remote_code)
        if cfg is not None:
            kw["config"] = cfg
        if load_in_4bit:
            kw["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16)
        else:
            kw["torch_dtype"] = "auto"
        self.lm = AutoModelForCausalLM.from_pretrained(model, **kw)
        if trust_remote_code:
            from .compat import patch_generation
            patch_generation(self.lm)        # before any PEFT wrapper delegates to it
        # Set if the repository's cache handling turns out to be unusable and
        # generation has to fall back to recomputing the prefix each step.
        self._no_cache = False
        if adapter_path:                     # load a LoRA adapter (fine-tuned run)
            from peft import PeftModel
            self.lm = PeftModel.from_pretrained(self.lm, adapter_path)
        self.lm.eval()
        self.max_tokens = max_tokens

    def complete(self, triple: Dict) -> str:
        return self.complete_batch([triple])[0]

    def complete_batch(self, triples) -> list:
        """Batched greedy generation -- the throughput path for GPU runs.

        Base-weight protocol: raw prefix / pre-query text. For an aligned
        (Instruct) model, set use_chat_template=True so prompts are wrapped in
        the model's chat template and its safety/refusal behavior is active --
        the condition under which a cross-lingual gap (positive CLMD) can arise.
        With left padding, the prompt occupies the first ``n_in`` columns for
        every row, so the generated continuation is the same slice across the batch.
        """
        if self.use_chat:
            texts = [self.tok.apply_chat_template(
                        t.get("messages") or [{"role": "user", "content": t["prompt"]}],
                        tokenize=False, add_generation_prompt=True) for t in triples]
        else:
            texts = [t["prompt"] for t in triples]
        enc = self.tok(texts, return_tensors="pt", padding=True).to(self.lm.device)
        gen = {"max_new_tokens": self.max_tokens, "do_sample": False,
               "num_beams": 1, "pad_token_id": self.tok.pad_token_id}
        with self.torch.no_grad():
            try:
                out = self.lm.generate(**enc, **gen,
                                       **({"use_cache": False} if self._no_cache else {}))
            except TypeError as e:
                if self._no_cache:
                    raise
                # A repository whose cache plumbing the installed transformers
                # no longer matches. Recomputing the prefix each step is slower
                # but always correct, and finishing beats a clean failure.
                print(f"  [compat] cached generation failed ({e}); "
                      f"falling back to use_cache=False for the rest of the run")
                self._no_cache = True
                out = self.lm.generate(**enc, **gen, use_cache=False)
        n_in = enc["input_ids"].shape[1]
        return [self.tok.decode(o[n_in:], skip_special_tokens=True) for o in out]


# --------------------------------------------------------------------------- #
# Factory
# --------------------------------------------------------------------------- #
def _accept(kwargs: Dict, backend: str, allowed) -> Dict:
    """Filter kwargs to what this backend takes, loudly.

    The filter exists so one call site can pass options that only some backends
    understand. Dropping silently is how ``trust_remote_code=True`` went missing
    on the way to HFLocalBackend and surfaced an hour later as a model-load
    failure, so anything discarded is now named.
    """
    keep = {k: v for k, v in kwargs.items() if k in allowed}
    dropped = sorted(set(kwargs) - set(keep))
    if dropped:
        print(f"  [make_backend] warning: {backend} ignores {dropped} "
              f"(accepts {sorted(allowed)})")
    return keep


def make_backend(backend: str, model: str, **kwargs) -> Backend:
    backend = backend.lower()
    if backend == "demo":
        return DemoBackend(model, seed=kwargs.get("seed", 20260524))
    if backend in ("openai_compat", "dashscope"):
        return OpenAICompatBackend(model, **_accept(
            kwargs, backend, ("base_url", "api_key", "api_key_env",
                              "max_tokens", "retries", "sleep")))
    if backend in ("hf_api", "hf"):
        return HFInferenceBackend(model, **_accept(
            kwargs, backend, ("api_key", "api_key_env",
                              "max_tokens", "retries", "sleep")))
    if backend in ("hf_local", "local", "transformers"):
        return HFLocalBackend(model, **_accept(
            kwargs, backend, ("max_tokens", "device", "load_in_4bit",
                              "adapter_path", "use_chat_template",
                              "trust_remote_code")))
    raise ValueError(f"unknown backend: {backend}")


if __name__ == "__main__":
    from .m1_generator import build_dataset
    from .m2_prompts import build_prompt_matrix
    ds = build_dataset(50)
    triples = build_prompt_matrix(ds, categories=["national_id"])
    be = DemoBackend("qwen2.5-7b")
    hits = sum(1 for t in triples if t["expected"] in be.complete(t))
    print(f"[demo/qwen2.5-7b] national_id rough hit rate = {hits/len(triples):.2f} "
          f"over {len(triples)} queries")
