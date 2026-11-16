"""How does a model get around the ablation? (proposal, Evaluation section)

bridge_ranks_all_positions: J-lens rank of the bridge at the tokens where it is
    likely computed. We use two readouts: the final prompt token, and the best
    rank over the question span (every token after "Fact:", which contains the
    description of the bridge). Run it with the original and the recomputed
    lens (swap js.lens) to see whether the concept moved.
collect_final_token: residuals at the final token per layer, for probes.
"""

import statistics

import torch

from evaluation.conditions import CONDITIONS
from jspace import lens_logits, token_variants
from model.inference import encode, forward_residuals


def question_start(js, prompt: str) -> int:
    """Index of the first token after the 'Fact:' style prefix, in jlens's encoding."""
    prefix = prompt.split(":", 1)[0] + ":" if ":" in prompt else ""
    # TODO(debt-001): positions index jlens's own encoding of the prompt.
    return js.lens_model.encode(prefix).shape[1] if prefix else 0


def bridge_ranks_all_positions(js, item, condition: str, rng) -> dict[str, dict[int, int]]:
    """{'final': {layer: rank}, 'question_best': {layer: rank}} under a condition."""
    ids = token_variants(js.lm.tokenizer, item.bridge)
    interventions = CONDITIONS[condition](js, item, encode(js.lm, [item.prompt]), rng)
    logits_by_layer = lens_logits(js, item.prompt, interventions)
    start = question_start(js, item.prompt)
    out = {"final": {}, "question_best": {}}
    for layer, logits in logits_by_layer.items():  # logits: [seq_len, vocab]
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
        enc = encode(js.lm, [item.prompt])
        interventions = CONDITIONS[condition](js, item, enc, rng_for(item))
        rows.append(_final_token_residuals(js, enc, interventions))
    return torch.stack(rows).cpu()


def _final_token_residuals(js, enc, interventions) -> torch.Tensor:
    """[n_layers, d] fp32 post-intervention residuals at the last position."""
    layers = list(range(js.lm.n_layers))
    resids = forward_residuals(js.lm, enc, layers, interventions)
    return torch.stack([resids[l][0, -1].float() for l in layers])
