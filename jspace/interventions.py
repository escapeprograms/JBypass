"""Residual-stream interventions in J-space. Each is a plain `model.hooks.Intervention`
(no hook code here): pass it to any `model.inference` function, `apply_interventions`,
or `jspace.readout`:

    logits = next_token_logits(js.lm, [prompt], [ProjectOut(js, vectors)])

All three edit the output of each listed layer at every real token of a row except its
first js.cfg.ablation_skip_first ones; left padding is never edited. Qwen's first
position is an attention sink with huge activations (jlens.fitting skips early positions
for the same reason); editing it breaks the model outright instead of removing concepts.
"""

import torch

from jspace.vectors import lens_activations
from model.hooks import HookContext
from model.inference import position_ids


class LayerIntervention:
    """Base class: at each of `layers`, replaces the residual with self.edit(h, layer),
    computed in fp32 and cast back to the residual's dtype, keeping the original values
    at padding and at each row's leading positions. Subclasses implement edit only.

    `edit` receives the residual from the first column any row may edit onward (all of
    it in a padded batch whose shortest row starts at column 0). Editing only that tail
    keeps an unpadded prompt's numbers bit-identical to editing h[:, skip_first:]; fp32
    matmuls over a different number of rows can round differently."""

    def __init__(self, js, layers: list[int]):
        self.js = js
        self.layers = layers
        self.skip_first = js.cfg.ablation_skip_first

    def edit(self, h: torch.Tensor, layer: int) -> torch.Tensor:
        raise NotImplementedError

    def __call__(self, resid: torch.Tensor, ctx: HookContext) -> torch.Tensor:
        keep = editable_positions(ctx.attention_mask, self.skip_first)
        start = first_editable_column(keep)
        edited = self.edit(resid[:, start:].float(), ctx.layer).to(resid.dtype)
        tail = torch.where(keep[:, start:, None], edited, resid[:, start:])
        return torch.cat([resid[:, :start], tail], dim=1)


def editable_positions(attention_mask: torch.Tensor, skip_first: int) -> torch.Tensor:
    """[batch, seq] bool: real tokens at or after the row's `skip_first`-th real token."""
    return attention_mask.bool() & (position_ids(attention_mask) >= skip_first)


def first_editable_column(keep: torch.Tensor) -> int:
    """First column where any row is editable (0 if none is: then nothing is kept edited)."""
    return int(keep.any(0).int().argmax())


def _project_out(h: torch.Tensor, Q: torch.Tensor) -> torch.Tensor:
    """Remove span(Q) from h. Q: [d, k] orthonormal, or [..., d, k] per position."""
    if Q.dim() == 2:
        return h - (h @ Q) @ Q.T
    coeffs = (Q.transpose(-1, -2) @ h.unsqueeze(-1))
    return h - (Q @ coeffs).squeeze(-1)


class ProjectOut(LayerIntervention):
    """Zero the residual's projection onto fixed J-lens vectors (e.g. the
    bridge entity's), per layer. vectors: {layer: [k, d]}. Joint orthogonal
    projection, so the residual ends with zero inner product with every vector."""

    def __init__(self, js, vectors: dict[int, torch.Tensor]):
        super().__init__(js, sorted(vectors))
        self.bases = {l: torch.linalg.qr(v.float().T).Q for l, v in vectors.items()}

    def edit(self, h, layer):
        return _project_out(h, self.bases[layer].to(h.device))


class TopKAblation(LayerIntervention):
    """Paper's J-space ablation: "at each token position, across a band of
    layers, we identify the k=10 most strongly activated J-lens vectors and zero
    out the residual stream's projection onto each". Tokens in exclude_mask
    (the clean pass's top output tokens) are never ablated.

    exclude_mask: [batch, seq_len, vocab] bool aligned with the full sequence, or None.
    edit() may get only the tail of the sequence, so the mask is aligned on the end."""

    def __init__(self, js, layers: list[int], k: int, exclude_mask: torch.Tensor | None):
        super().__init__(js, layers)
        self.k = k
        self.exclude_mask = exclude_mask

    def edit(self, h, layer):
        W_U = self.js.lm.unembedding_matrix
        acts = lens_activations(self.js.lens, W_U, h, layer)  # [B, T, V]
        if self.exclude_mask is not None:
            mask = self.exclude_mask[:, -h.shape[1]:].to(acts.device)
            acts = acts.masked_fill(mask, float("-inf"))
        top = acts.topk(self.k, dim=-1).indices  # [B, T, k]
        J = self.js.lens.jacobians[layer].to(h.device)
        vecs = W_U[top].float() @ J  # [B, T, k, d]
        Q = torch.linalg.qr(vecs.transpose(-1, -2)).Q  # [B, T, d, k]
        return _project_out(h, Q)


class LensSwap(LayerIntervention):
    """Paper's swap (Methods, "Writing"): V = [v_s v_t], c = pinv(V) h,
    h <- h + alpha * V (swap(c) - c). Leaves the part of h orthogonal to
    span{v_s, v_t} untouched. source/target: {layer: [d]}."""

    def __init__(self, js, source: dict[int, torch.Tensor], target: dict[int, torch.Tensor], alpha: float = 1.0):
        super().__init__(js, sorted(source))
        self.V = {l: torch.stack([source[l], target[l]], dim=1).float() for l in source}
        self.V_pinv = {l: torch.linalg.pinv(V) for l, V in self.V.items()}  # [2, d]
        self.alpha = alpha

    def edit(self, h, layer):
        V = self.V[layer].to(h.device)  # [d, 2]
        c = h @ self.V_pinv[layer].to(h.device).T  # [..., 2]
        delta = c.flip(-1) - c
        return h + self.alpha * (delta @ V.T)
