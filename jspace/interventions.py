"""Residual-stream interventions in J-space, applied as forward hooks on the
decoder layers. Use as context managers:

    with ProjectOut(js, vectors):
        logits = js.hf_model(input_ids).logits

All three edit the output of each listed layer at every token position except
the first js.cfg.ablation_skip_first ones. Qwen's first position is an attention
sink with huge activations (jlens.fitting skips early positions for the same
reason); editing it breaks the model outright instead of removing concepts.
"""

import torch

from jspace.vectors import lens_activations


class LayerIntervention:
    """Base class: hooks model.layers[l] for l in layers and replaces the
    residual with self.edit(h, l). Subclasses implement edit only."""

    def __init__(self, js, layers: list[int]):
        self.js = js
        self.layers = layers
        self._handles: list = []

    def edit(self, h: torch.Tensor, layer: int) -> torch.Tensor:
        raise NotImplementedError

    def _hook(self, layer: int):
        skip = self.js.cfg.ablation_skip_first

        def hook(module, inputs, output):
            h = output if torch.is_tensor(output) else output[0]
            edited = self.edit(h[:, skip:].float(), layer).to(h.dtype)
            new = torch.cat([h[:, :skip], edited], dim=1)
            return new if torch.is_tensor(output) else (new, *output[1:])

        return hook

    def __enter__(self):
        for l in self.layers:
            block = self.js.model.layers[l]
            self._handles.append(block.register_forward_hook(self._hook(l)))
        return self

    def __exit__(self, *exc):
        for handle in self._handles:
            handle.remove()
        self._handles = []


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

    exclude_mask: [1, seq_len, vocab] bool, or None. Single-prompt (batch 1) use."""

    def __init__(self, js, layers: list[int], k: int, exclude_mask: torch.Tensor | None):
        super().__init__(js, layers)
        self.k = k
        self.exclude_mask = exclude_mask

    def edit(self, h, layer):
        acts = lens_activations(self.js.lens, self.js.W_U, h, layer)  # [B, T, V]
        if self.exclude_mask is not None:  # h may skip leading positions: align on the end
            mask = self.exclude_mask[:, -h.shape[1]:].to(acts.device)
            acts = acts.masked_fill(mask, float("-inf"))
        top = acts.topk(self.k, dim=-1).indices  # [B, T, k]
        J = self.js.lens.jacobians[layer].to(h.device)
        vecs = self.js.W_U[top].float() @ J  # [B, T, k, d]
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
