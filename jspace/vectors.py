"""J-lens vectors and activations.

Paper (Methods, "The Jacobian Lens"): "We refer to the rows of W_U J_l as the
Jacobian lens (J-lens) vectors at layer l". So the vector for token v at layer l
is W_U[v] @ J_l (a direction in layer-l residual space), and its activation on a
residual h is W_U[v] @ J_l @ h. The final norm is not part of the vector.
"""

import torch


def workspace_band(n_layers: int, start: float, end: float) -> list[int]:
    """Layer indices between fractions start and end of depth (inclusive)."""
    return list(range(round(start * n_layers), round(end * n_layers) + 1))


def band_prefix(band: list[int], fraction: float) -> list[int]:
    """First `fraction` of the band's layers (at least one)."""
    return band[: max(1, round(fraction * len(band)))]


def token_variants(tokenizer, word: str) -> list[int]:
    """Ids of every single-token spelling of word: with/without leading space,
    original/lower/capitalized case. The J-lens only reads single-token concepts."""
    spellings = [word, word.lower(), word.capitalize()]
    candidates = [prefix + s for s in spellings for prefix in (" ", "")]
    ids: list[int] = []
    for text in candidates:
        tokens = tokenizer(text, add_special_tokens=False).input_ids
        if len(tokens) == 1 and tokens[0] not in ids:
            ids.append(tokens[0])
    return ids


def lens_vectors(
    lens, W_U: torch.Tensor, token_ids: list[int], layers: list[int]
) -> dict[int, torch.Tensor]:
    """{layer: [len(token_ids), d_model]} fp32 J-lens vectors (rows of W_U J_l)."""
    rows = W_U[token_ids].float()
    return {l: rows @ lens.jacobians[l].to(rows.device) for l in layers}


def lens_activations(lens, W_U: torch.Tensor, h: torch.Tensor, layer: int) -> torch.Tensor:
    """Activation of every J-lens vector on residuals h [..., d] -> [..., vocab]."""
    J = lens.jacobians[layer].to(h.device)
    return (h.float() @ J.T) @ W_U.float().T
