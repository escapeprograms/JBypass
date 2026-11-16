"""Read concepts out of the residual stream with the official jlens lens.apply
(lens(h) = unembed(J_l h), unembed = final norm + LM head), optionally under
interventions."""

from collections.abc import Sequence

import torch

from model.hooks import Intervention, apply_interventions


def lens_logits(js, prompt: str, interventions: Sequence[Intervention] = (),
                layers: list[int] | None = None,
                positions: list[int] | None = None) -> dict[int, torch.Tensor]:
    """{layer: [n_positions, vocab]} J-lens logits of one prompt, with the interventions
    applied during jlens's forward pass. Positions index jlens's own encoding of the
    prompt (all positions when None); layers default to every fitted layer."""
    layers = js.lens.source_layers if layers is None else layers
    with apply_interventions(js.lm, interventions, _unpadded_mask(js, prompt)):
        logits, _, _ = js.lens.apply(js.lens_model, prompt, layers=layers, positions=positions)
    return logits


def token_ranks(js, prompt: str, token_ids: list[int], position: int = -1,
                layers: list[int] | None = None,
                interventions: Sequence[Intervention] = ()) -> dict[int, int]:
    """Best (lowest) rank among token_ids in the J-lens readout at one position,
    per layer. Rank 0 = top-1."""
    logits = lens_logits(js, prompt, interventions, layers, [position])
    return {l: _best_rank(row[0], token_ids) for l, row in logits.items()}


def readout_layers(js) -> list[int]:
    """Layers residual_logits can read: the fitted layers plus the model's final layer."""
    return sorted(set(js.lens.source_layers) | {js.lm.n_layers - 1})


def residual_logits(js, resid: torch.Tensor, layer: int) -> torch.Tensor:
    """fp32 [..., vocab] J-lens logits of residuals [..., d] taken at `layer`, as in
    jlens.vis.compute_slice: transport by J_l, then the lens model's unembed (final norm +
    LM head, cast to the head's dtype). The final layer needs no Jacobian (J = I), so its
    row is the model's own output. Raises ValueError for any other unfitted layer."""
    final = js.lm.n_layers - 1
    if layer in js.lens.jacobians:
        resid = js.lens.transport(resid.float(), layer)
    elif layer != final:
        raise ValueError(f"layer {layer} has no fitted Jacobian and is not the final layer {final}")
    return js.lens_model.unembed(resid).float()


def _unpadded_mask(js, prompt: str) -> torch.Tensor:
    """All-ones attention mask shaped like jlens's encoding of the prompt (lens.apply runs
    one unpadded prompt)."""
    # TODO(debt-001): lens.apply tokenises the prompt itself; this mirrors its encoding.
    return torch.ones_like(js.lens_model.encode(prompt))


def _best_rank(logits: torch.Tensor, token_ids: list[int]) -> int:
    best = logits[token_ids].max()
    return int((logits > best).sum())
