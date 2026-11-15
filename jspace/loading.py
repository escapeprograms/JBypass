"""Load the model and its J-lens into one bundle. All network and file access
for the jspace module lives here."""

from dataclasses import dataclass

import jlens
import torch
import transformers

from jspace.config import JSpaceConfig
from jspace.vectors import workspace_band


@dataclass
class JSpace:
    """Everything the interventions and readouts need.

    hf_model: the HuggingFace *ForCausalLM (call it for logits; hooks on its
        layers apply because LensModel shares the same modules).
    model: jlens LensModel wrapper over hf_model (used by lens.apply).
    lens: fitted jlens.JacobianLens, J_l per layer.
    W_U: unembedding matrix [vocab, d_model].
    band: workspace layer indices.
    """

    hf_model: torch.nn.Module
    model: jlens.HFLensModel
    tokenizer: object
    lens: jlens.JacobianLens
    W_U: torch.Tensor
    band: list[int]
    cfg: JSpaceConfig

    @property
    def device(self) -> torch.device:
        return self.W_U.device


def load_jspace(cfg: JSpaceConfig = JSpaceConfig(), adapter_path: str | None = None) -> JSpace:
    """Load model (optionally merging a LoRA adapter) and the lens."""
    hf_model = transformers.AutoModelForCausalLM.from_pretrained(
        cfg.model_name, dtype=getattr(torch, cfg.dtype)
    ).to(cfg.device)
    if adapter_path is not None:
        hf_model = _merge_adapter(hf_model, adapter_path)
    tokenizer = transformers.AutoTokenizer.from_pretrained(cfg.model_name)
    model = jlens.from_hf(hf_model, tokenizer)
    lens = load_lens(cfg)
    W_U = hf_model.get_output_embeddings().weight
    band = workspace_band(model.n_layers, cfg.band_start, cfg.band_end)
    return JSpace(hf_model, model, tokenizer, lens, W_U, band, cfg)


def load_lens(cfg: JSpaceConfig) -> jlens.JacobianLens:
    if cfg.lens_path is not None:
        return jlens.JacobianLens.load(cfg.lens_path)
    return jlens.JacobianLens.from_pretrained(
        cfg.lens_repo, filename=cfg.lens_file, revision=cfg.lens_revision
    )


def _merge_adapter(hf_model: torch.nn.Module, adapter_path: str) -> torch.nn.Module:
    """Merge LoRA weights so hooks see the plain module tree (for Mctrl / Mexp)."""
    from peft import PeftModel

    merged = PeftModel.from_pretrained(hf_model, adapter_path).merge_and_unload()
    return merged.eval()
