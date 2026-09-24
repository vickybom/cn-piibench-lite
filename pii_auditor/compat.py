# -*- coding: utf-8 -*-
"""Shims for model repositories whose remote code predates current transformers.

Families that ship their own modelling code on the Hub - InternLM2 among them -
are pinned to whatever transformers API existed when that code was written. The
library moves; the repository does not. Loading such a model on a current
transformers therefore fails inside the *model's* code rather than ours, and the
fix has to be applied to the config before the model is constructed.

Only used when ``trust_remote_code`` is on, so nothing on the Qwen path changes.
"""
from __future__ import annotations

from typing import Any


def patched_config(model_name: str, trust_remote_code: bool = False) -> Any | None:
    """Config for ``model_name`` with legacy-incompatible fields repaired.

    Returns None when no repair is needed, so the caller can omit ``config=``
    and let ``from_pretrained`` behave exactly as before.

    The repair currently covers one case, which is the one that bites:

    transformers 4.45 renamed the RoPE scaling discriminator from ``type`` to
    ``rope_type`` and began populating ``rope_scaling`` by default. InternLM2's
    ``modeling_internlm2.py`` still reads ``config.rope_scaling["type"]`` and
    only branches on ``"linear"`` or ``"dynamic"``, raising for anything else.
    A model that declares *no* scaling therefore arrives with a populated dict
    whose key the old code cannot find, and dies on ``KeyError: 'type'``.

    So: a legacy scaling mode is renamed back to the key the old code reads, and
    the no-op ``default`` mode is turned back into ``None``, which is what that
    code expects to mean "no scaling".
    """
    if not trust_remote_code:
        return None
    from transformers import AutoConfig

    cfg = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
    rs = getattr(cfg, "rope_scaling", None)
    if not isinstance(rs, dict):
        return None
    if "type" in rs:
        return None                     # already in the shape the old code wants

    kind = rs.get("rope_type")
    if kind in ("linear", "dynamic"):
        rs["type"] = kind
        print(f"  [compat] rope_scaling: rope_type={kind!r} -> also set type={kind!r}")
    else:
        cfg.rope_scaling = None
        print(f"  [compat] rope_scaling: {rs} is a no-op for this model, "
              f"set to None so the repository's older code takes its "
              f"'no scaling' branch")
    return cfg


def _prepare_inputs(self, input_ids, past_key_values=None, attention_mask=None,
                    inputs_embeds=None, **kwargs):
    """Cache-API-current replacement for a repository's own version.

    The classic decoder-only contract: trim the prompt to the tokens the cache
    has not seen, rebuild position ids from the padding mask, and hand back only
    the keys the model's ``forward`` actually accepts. It reads cache length
    through ``get_seq_length()``, which has survived every rename, instead of
    the ``get_max_length`` / ``get_max_cache_shape`` pair that has not.
    """
    if past_key_values is not None:
        if hasattr(past_key_values, "get_seq_length"):
            past_len = past_key_values.get_seq_length()
        else:                                   # legacy tuple-of-tuples cache
            past_len = past_key_values[0][0].shape[2]
        past_len = int(past_len)
        if input_ids.shape[1] > past_len:
            input_ids = input_ids[:, past_len:]
        else:
            input_ids = input_ids[:, -1:]

    position_ids = kwargs.get("position_ids")
    if attention_mask is not None and position_ids is None:
        position_ids = attention_mask.long().cumsum(-1) - 1
        position_ids.masked_fill_(attention_mask == 0, 1)
        if past_key_values is not None:
            position_ids = position_ids[:, -input_ids.shape[1]:]

    return {"input_ids": input_ids,
            "position_ids": position_ids,
            "past_key_values": past_key_values,
            "use_cache": kwargs.get("use_cache", True),
            "attention_mask": attention_mask}


def patch_generation(model) -> bool:
    """Swap out a remote repository's stale generation-input preparation.

    Remote-code classes are namespaced under ``transformers_modules``, which is
    how a repository-supplied model is told apart from a library-native one.
    Only those get patched, so Qwen and Yi keep the library's own path.

    InternLM2's version reads the cache through ``get_max_length()``. On a
    transformers new enough to alias that to ``get_max_cache_shape()`` it gets a
    shape *tuple* back and dies on ``tuple + int`` the first time generation
    needs the cache - after the model has loaded, which is the expensive place
    to find out.
    """
    cls = type(model)
    if not getattr(cls, "__module__", "").startswith("transformers_modules"):
        return False
    import types as _types

    model.prepare_inputs_for_generation = _types.MethodType(_prepare_inputs, model)
    print(f"  [compat] {cls.__name__}.prepare_inputs_for_generation replaced with "
          f"a cache-API-current implementation")
    return True
