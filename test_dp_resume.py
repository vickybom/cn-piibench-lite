#!/usr/bin/env python
"""Does DP-SGD checkpoint/resume actually reproduce an uninterrupted run?

Worth ten seconds before committing eight hours. A checkpoint that restores the
weights but not the optimizer moments or the RNG streams still "works" - it
just silently continues on a different trajectory, and the epsilon being
claimed no longer describes the model that comes out.

Runs on a toy module on the CPU: no GPU, no corpus, no downloads.

    python test_dp_resume.py
"""
from __future__ import annotations

import sys
import tempfile
import warnings
from pathlib import Path

import torch
import torch.nn as nn
import peft
from peft import LoraConfig, get_peft_model

sys.path.insert(0, str(Path(__file__).parent))
from pii_auditor.defenses import dp_sgd_finetune  # noqa: E402

warnings.filterwarnings("ignore")

# Colab ships torchao 0.10.0 and peft 0.19 wants >0.16, so peft's LoRA
# dispatcher raises the moment it is offered a plain nn.Linear. The real runs
# never reach that dispatcher: the base model is loaded in 4-bit, the targets
# are bitsandbytes Linear4bit, and the bnb dispatcher matches first - which is
# why the undefended adapter trained without complaint in this same runtime.
# Only this toy uses an unquantized Linear. Declaring the backend absent for
# this process is narrower than upgrading torchao, which would drag torch with
# it and make every condition after it incomparable with the ones before.
for _mod in ("peft.import_utils", "peft.tuners.lora.torchao"):
    try:
        __import__(_mod)
        setattr(sys.modules[_mod], "is_torchao_available", lambda: False)
    except ImportError:
        pass
TEXTS = [f"record {i}" for i in range(16)]


class Toy(nn.Module):
    """Small enough to train in a second, real enough to have LoRA targets."""

    def __init__(self):
        super().__init__()
        self.emb = nn.Embedding(64, 32)
        self.q_proj = nn.Linear(32, 32)
        self.out = nn.Linear(32, 64)

    def forward(self, input_ids=None, labels=None, **kw):
        h = self.out(self.q_proj(self.emb(input_ids)))
        loss = nn.functional.cross_entropy(h.view(-1, 64), labels.view(-1))
        return type("Out", (), {"loss": loss})()


class Tok:
    """Stands in for the tokenizer: fixed-length ids, drawn deterministically
    from the same RNG stream the real data order uses, so a resumed run sees
    the same sequence a continuous one would."""

    def __call__(self, xs, **kw):
        class Enc(dict):
            def to(self, dev):
                # actually move, so the toy exercises the same device the noise
                # is drawn on - a no-op here would leave the ids on the CPU and
                # the GPU path would never be tested
                return Enc({k: v.to(dev) for k, v in self.items()})
        return Enc(input_ids=torch.randint(0, 64, (1, 8)))


# On a GPU box this has to run on the GPU. The noise is drawn with
# torch.normal(device=...), so on CUDA it comes from the CUDA generator, and
# whether torch.cuda.set_rng_state_all restores that stream is exactly the
# thing a CPU-only test cannot tell you.
DEV = "cuda" if torch.cuda.is_available() else "cpu"


def build():
    torch.manual_seed(0)                      # identical initialisation
    return get_peft_model(Toy(), LoraConfig(r=4, lora_alpha=8,
                                            target_modules=["q_proj"])).to(DEV)


def lora(m):
    return {k: v for k, v in m.state_dict().items() if "lora" in k}


def main():
    d = Path(tempfile.mkdtemp())
    ok = []
    print(f"running on {DEV}"
          + ("" if DEV == "cuda" else
             "   (no GPU here: the CUDA generator's restore is not exercised, "
             "so rerun this on the GPU box before trusting it there)"))

    print("=" * 68)
    print("A. four epochs, uninterrupted")
    print("=" * 68)
    torch.manual_seed(7)
    m_full, h_full = dp_sgd_finetune(build(), Tok(), TEXTS, epochs=4,
                                     batch_size=8, ckpt_dir=d / "full")

    print("\n" + "=" * 68)
    print("B. two epochs, then a fresh process resumes to four")
    print("=" * 68)
    torch.manual_seed(7)
    dp_sgd_finetune(build(), Tok(), TEXTS, epochs=2, batch_size=8,
                    ckpt_dir=d / "cut")
    print("   ---- new model, new optimizer, resumed from the checkpoint ----")
    m_res, h_res = dp_sgd_finetune(build(), Tok(), TEXTS, epochs=4,
                                   batch_size=8, ckpt_dir=d / "cut")

    a, b = lora(m_full), lora(m_res)
    worst = max(float((a[k] - b[k]).abs().max()) for k in a)
    ok.append(("resumed weights match the uninterrupted run",
               worst < 1e-6, f"largest difference over {len(a)} tensors {worst:.2e}"))
    ok.append(("loss history survives the restart",
               len(h_res) == len(h_full), f"{len(h_res)} steps vs {len(h_full)}"))
    ok.append(("history values match",
               all(abs(x["loss"] - y["loss"]) < 1e-5
                   for x, y in zip(h_res, h_full)), ""))

    # A checkpoint carries an epsilon. Resuming it under a different noise
    # multiplier would produce a model whose privacy guarantee is neither the
    # old one nor the new one, and nothing downstream would show it.
    try:
        dp_sgd_finetune(build(), Tok(), TEXTS, epochs=4, batch_size=8,
                        noise_multiplier=2.0, ckpt_dir=d / "cut")
        ok.append(("a mechanism mismatch is refused", False, "it was accepted"))
    except AssertionError as e:
        ok.append(("a mechanism mismatch is refused", True, str(e)[:60]))

    # Nothing should change for callers that do not ask for checkpointing.
    torch.manual_seed(7)
    m_none, _ = dp_sgd_finetune(build(), Tok(), TEXTS, epochs=4, batch_size=8)
    c = lora(m_none)
    ok.append(("unchanged when no ckpt_dir is given",
               all(torch.allclose(a[k], c[k], atol=1e-6) for k in a), ""))

    print("\n" + "=" * 68)
    for name, good, detail in ok:
        print(f"  {'OK ' if good else 'BAD'} {name:<46}{detail}")
    bad = [n for n, g, _ in ok if not g]
    print("\n" + ("RESUME IS EXACT - safe to rely on"
                  if not bad else f"{len(bad)} FAILED: {bad}"))
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
