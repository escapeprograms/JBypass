"""How does a model get around the ablation? (proposal, Evaluation section)

bridge_ranks_all_positions: J-lens rank of the bridge at the tokens where it is
    likely computed. We use two readouts: the final prompt token, and the best
    rank over the question span (every token after "Fact:", which contains the
    description of the bridge). Run it with the original and the recomputed
    lens (swap js.lens) to see whether the concept moved.
collect_final_token: residuals at the final token per layer, for probes.
"""

import contextlib
import statistics

import torch
from jlens import ActivationRecorder

from evaluation.conditions import CONDITIONS
from jspace import token_variants


def question_start(js, prompt: str) -> int:
    """Index of the first token after the 'Fact:' style prefix."""
    prefix = prompt.split(":", 1)[0] + ":" if ":" in prompt else ""
    return js.model.encode(prefix).shape[1] if prefix else 0


def bridge_ranks_all_positions(js, item, condition: str, rng) -> dict[str, dict[int, int]]:
    """{'final': {layer: rank}, 'question_best': {layer: rank}} under a condition."""
    ids = token_variants(js.tokenizer, item.bridge)
    intervention = CONDITIONS[condition](js, item, item.prompt, rng)
    with intervention or contextlib.nullcontext():
        lens_logits, _, _ = js.lens.apply(js.model, item.prompt)
    start = question_start(js, item.prompt)
    out = {"final": {}, "question_best": {}}
    for layer, logits in lens_logits.items():  # logits: [seq_len, vocab]
        ranks = (logits > logits[:, ids].max(-1, keepdim=True).values).sum(-1)
        out["final"][layer] = int(ranks[-1])
        out["question_best"][layer] = int(ranks[start:].min())
    return out


def summarize_ranks(per_item: list[dict], k: int, band: list[int]) -> dict:
    """Per readout: pass@k over all layers (paper's metric), pass@k within the
    band (where ablations act), and the median rank per layer."""
    summary = {}
    for readout in ("final", "question_best"):
        rows = [r[readout] for r in per_item]
        layers = sorted(rows[0])
        summary[readout] = {
            "pass_at_k": sum(min(r.values()) < k for r in rows) / len(rows),
            "band_pass_at_k": sum(min(r[l] for l in band if l in r) < k for r in rows) / len(rows),
            "median_rank_by_layer": {l: statistics.median(r[l] for r in rows) for l in layers},
        }
    return summary


@torch.no_grad()
def collect_final_token(js, items, condition: str, rng_for) -> torch.Tensor:
    """[n_items, n_layers, d] residuals at the final prompt token under a condition."""
    rows = []
    for item in items:
        intervention = CONDITIONS[condition](js, item, item.prompt, rng_for(item))
        rows.append(_final_token_residuals(js, item.prompt, intervention))
    return torch.stack(rows).cpu()


def _final_token_residuals(js, prompt, intervention) -> torch.Tensor:
    """The intervention must be entered before the recorder: hooks run in
    registration order, so the recorder then sees the edited residual."""
    layers = list(range(js.model.n_layers))
    with intervention or contextlib.nullcontext(), \
            ActivationRecorder(js.model.layers, at=layers) as rec:
        js.model.forward(js.model.encode(prompt))
    return torch.stack([rec.activations[l][0, -1].float() for l in layers])
