"""Adapter lifecycle: attach, save, load, and temporarily disable a LoRA adapter.

PEFT edits the original model's modules in place, so `lm.blocks` (and any hooks on
them) stay valid. Adapter weights are fp32 over a bf16 base (PEFT's
autocast_adapter_dtype default).
"""

import json
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

from peft import LoraConfig, PeftModel, get_peft_model

from model.config import LoraSettings
from model.loading import LanguageModel

PROFILE_FILE = "jbypass_profile.json"   # written beside the adapter weights


def attach_lora(lm: LanguageModel, settings: LoraSettings) -> None:
    """Wrap the model with a fresh trainable adapter on the profile's target modules;
    the base weights are frozen."""
    config = LoraConfig(
        r=settings.r,
        lora_alpha=settings.alpha,
        lora_dropout=settings.dropout,
        target_modules=list(lm.profile.lora_target_modules),
        task_type="CAUSAL_LM",
    )
    lm.hf_model = get_peft_model(lm.hf_model, config)


def save_adapter(lm: LanguageModel, path: str | Path) -> None:
    """Save the adapter weights plus the name of the profile they were trained on."""
    path = Path(path)
    lm.hf_model.save_pretrained(path)
    (path / PROFILE_FILE).write_text(json.dumps({"profile": lm.profile.name}), encoding="utf-8")


def saved_profile(path: str | Path) -> str:
    """Name of the profile a saved adapter was trained on (from its PROFILE_FILE)."""
    return json.loads((Path(path) / PROFILE_FILE).read_text(encoding="utf-8"))["profile"]


def load_adapter(lm: LanguageModel, path: str | Path) -> None:
    """Load a saved adapter (frozen, for inference) onto a model without one.
    Raises ValueError if it was saved from a different profile."""
    path = Path(path)
    saved = saved_profile(path)
    if saved != lm.profile.name:
        raise ValueError(f"adapter at {path} is for {saved!r}, but the loaded model is {lm.profile.name!r}")
    lm.hf_model = PeftModel.from_pretrained(lm.hf_model, path)


@contextmanager
def adapter_disabled(lm: LanguageModel) -> Iterator[None]:
    """Run the exact base model inside the block (e.g. base-vs-tuned J-lens)."""
    with lm.hf_model.disable_adapter():
        yield
