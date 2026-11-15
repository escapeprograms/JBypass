"""LanguageModel: the HF model + tokenizer + profile, and the only code that uses
model-specific attribute paths. `load_model` does the disk/network loading;
`LanguageModel(...)` wraps an already-built model (used by tests)."""

from collections.abc import Sequence

import torch
from torch import nn
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel, PreTrainedTokenizerBase

from jbypass.model.config import ModelProfile


class LanguageModel:
    """A causal LM described by a ModelProfile.

    `hf_model` is what forward passes call; it becomes a PeftModel after
    `attach_lora`. Blocks and the final norm are resolved on the original model,
    whose modules PEFT edits in place, so they stay valid after LoRA is attached.
    """

    def __init__(self, hf_model: PreTrainedModel, tokenizer: PreTrainedTokenizerBase, profile: ModelProfile):
        tokenizer.padding_side = "left"   # the tokenizer does not set it; batching relies on left padding
        self.profile = profile
        self.tokenizer = tokenizer
        self.hf_model: nn.Module = hf_model
        self._base = hf_model
        self._blocks = hf_model.get_submodule(profile.blocks_path)
        self._final_norm = hf_model.get_submodule(profile.final_norm_path)

    @property
    def blocks(self) -> Sequence[nn.Module]:
        """Decoder blocks in order; hook point "layer i" is the output of blocks[i]."""
        return self._blocks

    @property
    def n_layers(self) -> int:
        return len(self._blocks)

    @property
    def d_model(self) -> int:
        return self._base.config.hidden_size

    @property
    def device(self) -> torch.device:
        return self._base.get_input_embeddings().weight.device

    @property
    def dtype(self) -> torch.dtype:
        return self._base.get_input_embeddings().weight.dtype

    @property
    def unembedding_matrix(self) -> torch.Tensor:
        """[vocab, d_model] weight of the output embedding (lm_head)."""
        return self._base.get_output_embeddings().weight

    def unembed(self, resid: torch.Tensor) -> torch.Tensor:
        """Logits from a residual: lm_head(final_norm(resid))."""
        return self._base.get_output_embeddings()(self._final_norm(resid))

    def token_ids(self, text: str) -> list[int]:
        """Plain encoding of `text`, without added special tokens."""
        return self.tokenizer(text, add_special_tokens=False).input_ids


def load_model(profile: ModelProfile) -> LanguageModel:
    """Load the profile's model (from the HF cache, downloading if needed) onto its device, in eval mode."""
    tokenizer = AutoTokenizer.from_pretrained(profile.hf_id)
    hf_model = AutoModelForCausalLM.from_pretrained(profile.hf_id, dtype=getattr(torch, profile.dtype))
    hf_model.to(profile.device).eval()
    return LanguageModel(hf_model, tokenizer, profile)
