"""Checks that the J-lens setup reproduces the paper before we trust it.

bridge_readout: is the bridge readable in the J-lens on the two-hop prompt?
    Paper's lens-eval metric: readout at the position right before the answer,
    pass@k = fraction of items whose min-over-layers rank is within top k.
swap_success: paper's swap (spider -> ant): swapping the bridge's lens
    coordinates for swap_to's should change the answer to swap_answer.
    Paper on Claude: 54-70% of two-hop prompts.
"""

import statistics

from evaluation.scoring import greedy_continuation, matches
from jspace import LensSwap, ProjectOut, lens_vectors, token_ranks, token_variants


def bridge_ranks(js, item, ablate_bridge: bool = False) -> dict[int, int]:
    """Best rank of the bridge per layer at the last prompt position."""
    ids = token_variants(js.tokenizer, item.bridge)
    if not ablate_bridge:
        return token_ranks(js, item.prompt, ids)
    with ProjectOut(js, lens_vectors(js.lens, js.W_U, ids, js.band)):
        return token_ranks(js, item.prompt, ids)


def bridge_readout(js, items, k: int, ablate_bridge: bool = False) -> dict:
    """pass@k over items, plus the median rank at each layer."""
    all_ranks = [bridge_ranks(js, it, ablate_bridge) for it in items]
    layers = sorted(all_ranks[0])
    return {
        "pass_at_k": sum(min(r.values()) < k for r in all_ranks) / len(all_ranks),
        "k": k,
        "median_rank_by_layer": {l: statistics.median(r[l] for r in all_ranks) for l in layers},
        "n": len(items),
    }


def swap_success(js, items, n_tokens: int) -> dict:
    """Among items the clean model solves, fraction whose answer becomes
    swap_answer after the lens swap bridge -> swap_to over the band."""
    trials = [_swap_trial(js, it, n_tokens) for it in items if it.swap_answers]
    solved = [t for t in trials if t is not None]
    return {"success": sum(solved) / max(1, len(solved)), "n_solved": len(solved), "n": len(trials)}


def _swap_trial(js, item, n_tokens: int) -> bool | None:
    """None if the clean model gets the item wrong; else whether the swap worked."""
    clean, _ = greedy_continuation(js, item.prompt, lambda text: None, n_tokens)
    if not matches(clean, item.answers):
        return None
    src = lens_vectors(js.lens, js.W_U, token_variants(js.tokenizer, item.bridge)[:1], js.band)
    tgt = lens_vectors(js.lens, js.W_U, token_variants(js.tokenizer, item.swap_to)[:1], js.band)
    swap = LensSwap(js, {l: v[0] for l, v in src.items()}, {l: v[0] for l, v in tgt.items()})
    swapped, _ = greedy_continuation(js, item.prompt, lambda text: swap, n_tokens)
    return matches(swapped, item.swap_answers)
