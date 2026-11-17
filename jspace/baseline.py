"""Reference statistics of the J-lens readout: per-token, per-layer mean and standard
deviation of the lens logit `h . v_t` over clean forward passes on reference texts.
`clamp.ReadoutClamp` caps each readout at mean + z * std, the edge of its normal range.

No file paths or dataset knowledge here: the texts and the ids to exclude come from the
caller, and `save_readout_stats` / `load_readout_stats` take explicit paths."""

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import torch

from jspace.interventions import editable_positions
from jspace.vectors import lens_activations
from model.inference import encode, forward_residuals


@dataclass(frozen=True)
class ReadoutStats:
    """layers: the layers with statistics (row order of mean/std).
    mean, std: [len(layers), vocab] fp32 on CPU. count: [vocab] positions averaged per
    token (the same at every layer). model: profile name. n_texts: reference texts used."""

    layers: list[int]
    mean: torch.Tensor
    std: torch.Tensor
    count: torch.Tensor
    model: str
    n_texts: int

    def caps(self, token_ids: Sequence[int], layers: Sequence[int], z: float) -> dict[int, torch.Tensor]:
        """{layer: [len(token_ids)]} mean + z * std. Raises for a layer without statistics
        or a token that never contributed (its statistics are undefined)."""
        missing = [l for l in layers if l not in self.layers]
        if missing:
            raise ValueError(f"readout stats have no layers {missing} (have {self.layers})")
        ids = torch.tensor(list(token_ids), dtype=torch.long)
        unseen = ids[self.count[ids] == 0].tolist()
        if unseen:
            raise ValueError(f"tokens {unseen} have no reference positions in the readout stats")
        rows = {l: self.layers.index(l) for l in layers}
        return {l: self.mean[r, ids] + z * self.std[r, ids] for l, r in rows.items()}


@torch.no_grad()
def collect_readout_stats(
    js, texts: Sequence[str], exclude_ids: Sequence[Sequence[int]], batch_size: int
) -> ReadoutStats:
    """Statistics over every editable position (real tokens after the first
    js.cfg.ablation_skip_first) of `texts`, at each band layer. exclude_ids[i]: tokens
    whose readout text i does not contribute to (e.g. the item's own bridge)."""
    W_U = js.lm.unembedding_matrix
    sums = _empty_sums(len(js.band), W_U.shape[0])
    for start in range(0, len(texts), batch_size):
        enc = encode(js.lm, texts[start:start + batch_size])
        excluded = _exclusion_mask(exclude_ids[start:start + batch_size], W_U.shape[0], js.lm.device)
        keep = editable_positions(enc.attention_mask, js.cfg.ablation_skip_first)
        m = keep[:, :, None] & ~excluded[:, None, :]  # [B, T, V]
        resids = forward_residuals(js.lm, enc, js.band)
        for row, layer in enumerate(js.band):
            _accumulate(lens_activations(js.lens, W_U, resids[layer], layer), m, sums, row)
        sums["count"] += m.sum((0, 1)).cpu()
    return _finish(sums, js.band, js.lm.profile.name, len(texts))


def _empty_sums(n_layers: int, vocab: int) -> dict[str, torch.Tensor]:
    """fp64 running sums on CPU: sum and sumsq [n_layers, vocab], count [vocab]."""
    return {
        "sum": torch.zeros(n_layers, vocab, dtype=torch.float64),
        "sumsq": torch.zeros(n_layers, vocab, dtype=torch.float64),
        "count": torch.zeros(vocab, dtype=torch.long),
    }


def _exclusion_mask(exclude_ids: Sequence[Sequence[int]], vocab: int, device) -> torch.Tensor:
    """[len(exclude_ids), vocab] bool, True at each row's excluded ids."""
    mask = torch.zeros(len(exclude_ids), vocab, dtype=torch.bool, device=device)
    for row, ids in enumerate(exclude_ids):
        mask[row, list(ids)] = True
    return mask


def _accumulate(acts: torch.Tensor, m: torch.Tensor, sums: dict[str, torch.Tensor], row: int) -> None:
    """Add the activations acts [B, T, V] where m is True to sums["sum"/"sumsq"][row]."""
    x = acts.double() * m
    sums["sum"][row] += x.sum((0, 1)).cpu()
    sums["sumsq"][row] += (x * x).sum((0, 1)).cpu()


def _finish(sums: dict[str, torch.Tensor], layers: list[int], model: str, n_texts: int) -> ReadoutStats:
    """Mean and std from the running sums. A token with count 0 gets mean 0 and std 0,
    and ReadoutStats.caps refuses it."""
    n = sums["count"].clamp(min=1).double()
    mean = sums["sum"] / n
    std = (sums["sumsq"] / n - mean ** 2).clamp(min=0).sqrt()
    return ReadoutStats(list(layers), mean.float(), std.float(), sums["count"], model, n_texts)


def save_readout_stats(stats: ReadoutStats, path: str | Path) -> None:
    """torch.save the fields of `stats` to `path`, creating its folder."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(vars(stats), path)


def load_readout_stats(path: str | Path, lm, band: Sequence[int]) -> ReadoutStats:
    """Stats saved by `save_readout_stats`. Raises ValueError if they were collected on
    another model, for another vocab size, or lack a band layer."""
    stats = ReadoutStats(**torch.load(path, map_location="cpu"))
    if stats.model != lm.profile.name:
        raise ValueError(f"readout stats are for model {stats.model!r}, not {lm.profile.name!r}")
    vocab = lm.unembedding_matrix.shape[0]
    if stats.mean.shape[1] != vocab:
        raise ValueError(f"readout stats vocab {stats.mean.shape[1]} does not match the model's {vocab}")
    missing = sorted(set(band) - set(stats.layers))
    if missing:
        raise ValueError(f"readout stats lack band layers {missing} (have {stats.layers})")
    return stats
