"""Read concepts out of the residual stream with the official jlens lens.apply
(lens(h) = unembed(J_l h), unembed = final norm + LM head)."""

import torch


def token_ranks(js, prompt: str, token_ids: list[int], position: int = -1,
                layers: list[int] | None = None) -> dict[int, int]:
    """Best (lowest) rank among token_ids in the J-lens readout at one position,
    per layer. Rank 0 = top-1. Runs inside any active intervention context."""
    layers = js.lens.source_layers if layers is None else layers
    lens_logits, _, _ = js.lens.apply(js.model, prompt, layers=layers, positions=[position])
    return {l: _best_rank(lens_logits[l][0], token_ids) for l in layers}


def _best_rank(logits: torch.Tensor, token_ids: list[int]) -> int:
    best = logits[token_ids].max()
    return int((logits > best).sum())
