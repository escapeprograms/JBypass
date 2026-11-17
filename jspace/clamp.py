"""ReadoutClamp: cap the J-lens readout of fixed tokens at the top of its normal range.

Why this exists: `interventions.ProjectOut` sets the readout `h . v_t` of the bridge's
J-lens vectors to exactly 0 at every edited position. That value never occurs in a clean
forward pass, so it is itself a signal: a model fine-tuned under ProjectOut learned to
exploit it ("readout pinned at 0 -> answer this way") and got more accurate with the
ablation on than off. The clamp applies the same minimum-norm edit with a different
target, `l_t <- min(l_t, mu_t + z * sigma_t)` (statistics from `baseline.ReadoutStats`):
it lowers a readout only where it is abnormally high, and leaves every position whose
readouts are already at or below their caps bit-identical to the clean run (the target
equals the readout, so the update is exactly zero).
"""

import torch

from jspace.interventions import LayerIntervention


def set_readout(h: torch.Tensor, V: torch.Tensor, V_pinv: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """Minimum-norm change to h [..., d] giving h @ V.T == targets [..., k].
    V: [k, d] (linearly independent rows), V_pinv = pinv(V): [d, k]. With zero targets
    this removes span(V), like interventions.ProjectOut."""
    return h + (targets - h @ V.T) @ V_pinv.T


class ReadoutClamp(LayerIntervention):
    """At each layer, lowers each readout h @ V.T that exceeds its cap to the cap.
    vectors: {layer: [k, d]} J-lens vectors; caps: {layer: [k]} (e.g. ReadoutStats.caps).
    Leaves the part of h orthogonal to span(V) untouched."""

    def __init__(self, js, vectors: dict[int, torch.Tensor], caps: dict[int, torch.Tensor]):
        super().__init__(js, sorted(vectors))
        self.V = {l: v.float() for l, v in vectors.items()}
        self.V_pinv = {l: torch.linalg.pinv(V) for l, V in self.V.items()}  # [d, k]
        self.caps = {l: caps[l].float() for l in vectors}

    def edit(self, h, layer):
        V = self.V[layer].to(h.device)
        readout = h @ V.T  # [..., k]
        targets = torch.minimum(readout, self.caps[layer].to(h.device))
        return set_readout(h, V, self.V_pinv[layer].to(h.device), targets)
