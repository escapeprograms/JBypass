"""J-lens readout of a token sequence as a jlens.vis.SliceData (tensors only, no HTML).

Same outputs as jlens.vis.compute_slice (layer_stride=1, no windowing, no display mask),
but on our own token ids and with interventions applied during the forward pass, so a
generated response (which may not round-trip through the tokenizer) reads out exactly.
Rows at ablated layers show the post-intervention residual."""

from collections.abc import Sequence

import numpy as np
import torch
from jlens.vis import SliceData

from evaluation.conditions import interventions_for_item
from evaluation.scoring import answer_token_ids
from jspace import readout_layers, residual_logits, single_token_ids, token_variants
from model.hooks import Intervention
from model.inference import Encoded, forward_residuals
from viz.run import RunResult

RANK_CHUNK = 256   # positions ranked per full-vocab sort (bounds the sort buffer, as in jlens)


def pinned_tokens(js, result: RunResult) -> list[int]:
    """Sorted ids of the bridge's single-token spellings (English variants and the single-token
    Chinese spellings) plus the answers' first tokens."""
    # TODO(debt): jlens pins at most 8 ids on load (slice_vis.html, id order); the rest stay tracked and pinnable by click
    tokenizer = js.lm.tokenizer
    bridge = set(token_variants(tokenizer, result.item.bridge))
    bridge_zh = set(single_token_ids(tokenizer, result.item.bridge_zh))
    return sorted(bridge | bridge_zh | answer_token_ids(tokenizer, result.answers))


def run_slice(js, result: RunResult, top_n: int, max_tracked: int | None) -> SliceData:
    """The readout of prompt + response up to the answer token (`result.shown_ids`) under the
    result's condition, rebuilt once on that sequence, with the bridge and answers pinned."""
    return run_slices(js, [js], result, top_n, max_tracked)[0]


@torch.no_grad()
def run_slices(js, readers: Sequence, result: RunResult, top_n: int,
               max_tracked: int | None) -> list[SliceData]:
    """run_slice once per reader (a JSpace on the same model, e.g. with another lens), from one
    forward pass: the condition's interventions are built from `js` (the original lens) and the
    residuals captured at the union of the readers' readout layers."""
    enc = shown_encoding(js, result)
    interventions = interventions_for_item(js, result.item, result.condition, result.seed)(enc)
    layers = sorted(set().union(*(readout_layers(r) for r in readers)))
    resids = forward_residuals(js.lm, enc, layers, interventions)
    pinned = pinned_tokens(js, result)
    return [slice_from_resids(r, enc, resids, top_n, max_tracked, pinned) for r in readers]


def shown_encoding(js, result: RunResult) -> Encoded:
    """One unpadded row of prompt + shown response ids on js's device."""
    ids = torch.tensor([result.prompt_ids + result.shown_ids], dtype=torch.long, device=js.lm.device)
    return Encoded(ids, torch.ones_like(ids))


@torch.no_grad()
def slice_from_ids(js, enc: Encoded, interventions: Sequence[Intervention], top_n: int,
                   max_tracked: int | None, pinned: Sequence[int]) -> SliceData:
    """SliceData of one unpadded row: top-n ids per (position, layer), the tracked tokens
    (jlens's frequency rule capped by max_tracked, plus `pinned`) and their full-vocab ranks."""
    resids = forward_residuals(js.lm, enc, readout_layers(js), interventions)
    return slice_from_resids(js, enc, resids, top_n, max_tracked, pinned)


@torch.no_grad()
def slice_from_resids(js, enc: Encoded, resids: dict[int, torch.Tensor], top_n: int,
                      max_tracked: int | None, pinned: Sequence[int]) -> SliceData:
    """slice_from_ids from residuals already captured for `enc` (at least at js's readout layers),
    read through js's lens."""
    layers = readout_layers(js)
    top_ids, vocab_size = _top_ids(js, resids, layers, top_n)
    tracked = _tracked_tokens(top_ids, max_tracked, pinned)
    ids = enc.input_ids[0].tolist()
    tokenizer = js.lm.tokenizer
    return SliceData(
        seq_len=len(ids), layers=layers, context_token_ids=ids,
        context_token_strs=[_token_str(tokenizer, t) for t in ids],
        top_ids=top_ids, top_ranks=_top_ranks(top_ids.shape),
        tracked_token_ids=tracked, rank_tensor=_rank_tensor(js, resids, layers, tracked),
        vocab_fragment=_vocab_fragment(tokenizer, top_ids, tracked, ids),
        vocab_size=vocab_size, pinned_token_ids=sorted(set(pinned)), ctx_offset=0,
    )


def replay_mismatches(slice_data: SliceData, result: RunResult) -> list[int]:
    """Indices i into shown_ids (= response_ids up to the answer token) where the final
    layer's top-1 at the position before response token i is not that token.
    Empty = the readout reproduces generation over the shown tokens."""
    start = len(result.prompt_ids) - 1
    final_top1 = slice_data.top_ids[:, -1, 0]   # layers are sorted; the last is the model's final layer
    return [i for i, tok in enumerate(result.shown_ids) if int(final_top1[start + i]) != tok]


def _layer_logits(js, resids: dict[int, torch.Tensor], layer: int) -> torch.Tensor:
    """fp32 [seq, vocab] J-lens logits of the single row at `layer`."""
    return residual_logits(js, resids[layer][0], layer)


def _top_ids(js, resids, layers: list[int], top_n: int) -> tuple[np.ndarray, int]:
    """[seq, n_layers, top_n] int32 top ids, one layer's logits at a time; and the vocab size."""
    seq_len = next(iter(resids.values())).shape[1]
    top_ids = np.zeros((seq_len, len(layers), top_n), dtype=np.int32)
    vocab_size = 0
    for i, layer in enumerate(layers):
        logits = _layer_logits(js, resids, layer)
        vocab_size = int(logits.shape[-1])
        top_ids[:, i] = logits.topk(top_n, dim=-1).indices.cpu().numpy()
        del logits
    return top_ids, vocab_size


def _top_ranks(shape: tuple[int, ...]) -> np.ndarray:
    """Rank of each top-n cell: 0..top_n-1 along the last axis."""
    return np.broadcast_to(np.arange(shape[-1], dtype=np.int32), shape).copy()


def _tracked_tokens(top_ids: np.ndarray, max_tracked: int | None, pinned: Sequence[int]) -> list[int]:
    """jlens's rule: score each top token by sum of 1/(rank+1) over cells, keep the best
    max_tracked (ties in first-seen order), add the pins, sort."""
    flat_ids = top_ids.ravel()
    flat_ranks = _top_ranks(top_ids.shape).ravel()
    score: dict[int, float] = {}
    for token_id, rank in zip(flat_ids.tolist(), flat_ranks.tolist(), strict=True):
        score[token_id] = score.get(token_id, 0.0) + 1.0 / (rank + 1)
    by_score = sorted(score, key=score.__getitem__, reverse=True)
    return sorted(set(by_score[:max_tracked]) | set(pinned))


def _rank_tensor(js, resids, layers: list[int], tracked: list[int]) -> np.ndarray:
    """[seq, n_layers, n_tracked] int32 full-vocab ranks of the tracked tokens (0 = top)."""
    seq_len = next(iter(resids.values())).shape[1]
    ranks = np.full((seq_len, len(layers), len(tracked)), -1, dtype=np.int32)
    for i, layer in enumerate(layers):
        logits = _layer_logits(js, resids, layer)
        ranks[:, i] = _ranks_of(logits, torch.tensor(tracked, dtype=torch.long, device=logits.device))
        del logits
    return ranks


def _ranks_of(logits: torch.Tensor, targets: torch.Tensor) -> np.ndarray:
    """[seq, n_targets] ranks of `targets` in each row of [seq, vocab] logits: position in a
    descending argsort (jlens.vis's tie rule), chunked over positions."""
    arange = torch.arange(logits.shape[-1], device=logits.device)
    chunks = []
    for start in range(0, logits.shape[0], RANK_CHUNK):
        order = logits[start:start + RANK_CHUNK].argsort(dim=-1, descending=True)
        full_rank = torch.empty_like(order).scatter_(1, order, arange.expand_as(order))
        chunks.append(full_rank[:, targets].cpu())
    return torch.cat(chunks).numpy()


def _token_str(tokenizer, token_id: int) -> str:
    return tokenizer.decode([token_id], clean_up_tokenization_spaces=False)


def _vocab_fragment(tokenizer, top_ids: np.ndarray, tracked: list[int], context: list[int]) -> dict[int, str]:
    """Display string of every id the page can show: top ids, tracked ids, context ids."""
    ids = {int(t) for t in np.unique(top_ids)} | set(tracked) | set(context)
    return {t: _token_str(tokenizer, t) for t in ids}
