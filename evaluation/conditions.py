"""Ablation conditions. Each builder takes (js, item, enc, rng) and returns a sequence
of `model.hooks.Intervention`s, `()` for the clean run. `enc` is the current token
sequence (`model.inference.Encoded`, one row). Add a new condition by writing a builder
and registering it in CONDITIONS. Training uses the same builders (finetune/ablated.py).

  none          clean model
  bridge        project out the bridge's J-lens vectors over the band   (the intervention)
  wrong_bridge  same for item.swap_to, another bridge of the same kind  (specificity control)
  bridge_clamp  lower the bridge's J-lens readout over the band to its reference level
                mu + z*sigma (readout statistics, z = JSpaceConfig.clamp_z) only where it
                exceeds it; other positions untouched                    (in-distribution ablation)
  wrong_bridge_clamp
                same clamp for item.swap_to                             (specificity control)
  random_tok    same number of J-lens vectors of random word tokens     (Mrand-style control)
  random_dir    same number of random Gaussian directions               (any-ablation control)
  topk_light / topk_medium / topk_heavy
                paper's ablation: top-k active J-lens vectors per position, excluding
                the clean pass's top-k output tokens, over the first third / two
                thirds / all of the band                                (positive control)

The clamp conditions need `js.readout_stats` (python -m evaluation.run_readout_stats);
NEEDS_STATS names them.
"""

import random
from collections.abc import Callable, Sequence

import torch

from evaluation.data_loading import TwoHopItem
from jspace import (
    JSpace, ProjectOut, ReadoutClamp, ReadoutStats, TopKAblation, band_prefix, lens_vectors, token_variants,
)
from jspace.config import JSpaceConfig
from model.hooks import Intervention
from model.inference import Encoded, InterventionsFor, forward_logits

Builder = Callable[[JSpace, TwoHopItem, Encoded, random.Random], Sequence[Intervention]]


def item_rng(seed, item: TwoHopItem, condition: str) -> random.Random:
    """The per-item random source for a condition; one seeding rule for eval and training."""
    return random.Random(f"{seed}-{item.name}-{condition}")


def interventions_for_item(js, item: TwoHopItem, condition: str, seed: int) -> InterventionsFor:
    """enc -> the condition's interventions on that sequence. A fresh item_rng on every
    call, so random ablations stay fixed across decode steps."""
    def interventions_for(enc: Encoded) -> Sequence[Intervention]:
        return CONDITIONS[condition](js, item, enc, item_rng(seed, item, condition))
    return interventions_for


def build_none(js, item, enc, rng):
    return ()


def build_bridge(js, item, enc, rng):
    return (_project_word(js, item.bridge),)


def build_wrong_bridge(js, item, enc, rng):
    return (_project_word(js, item.swap_to),)


def build_bridge_clamp(js, item, enc, rng):
    return (_clamp_word(js, item.bridge),)


def build_wrong_bridge_clamp(js, item, enc, rng):
    return (_clamp_word(js, item.swap_to),)


def build_random_tok(js, item, enc, rng):
    n = len(token_variants(js.lm.tokenizer, item.bridge))
    ids = _random_word_tokens(js.lm.tokenizer, n, rng)
    return (ProjectOut(js, lens_vectors(js.lens, js.lm.unembedding_matrix, ids, js.band)),)


def build_random_dir(js, item, enc, rng):
    n = len(token_variants(js.lm.tokenizer, item.bridge))
    gen = torch.Generator().manual_seed(rng.randrange(2**31))
    d = js.lm.d_model
    return (ProjectOut(js, {l: torch.randn(n, d, generator=gen) for l in js.band}),)


def make_topk_builder(fraction: float) -> Builder:
    def build_topk(js, item, enc, rng):
        exclude = _clean_top_tokens_mask(js, enc, js.cfg.ablation_exclude_top_k)
        layers = band_prefix(js.band, fraction)
        return (TopKAblation(js, layers, js.cfg.ablation_top_k, exclude),)
    return build_topk


CONDITIONS: dict[str, Builder] = {
    "none": build_none,
    "bridge": build_bridge,
    "wrong_bridge": build_wrong_bridge,
    "bridge_clamp": build_bridge_clamp,
    "wrong_bridge_clamp": build_wrong_bridge_clamp,
    "random_tok": build_random_tok,
    "random_dir": build_random_dir,
}
# TODO(debt): strengths are read from the default JSpaceConfig at import time.
for _name, _fraction in JSpaceConfig().ablation_strengths:
    CONDITIONS[f"topk_{_name}"] = make_topk_builder(_fraction)

# Conditions whose builders read js.readout_stats.
NEEDS_STATS = frozenset({"bridge_clamp", "wrong_bridge_clamp"})


def require_stats(js) -> ReadoutStats:
    """js.readout_stats, or a ValueError saying how to get them."""
    if js.readout_stats is None:
        raise ValueError("this condition needs readout statistics: run "
                         "`python -m evaluation.run_readout_stats` or pass --stats_path")
    return js.readout_stats


def _word_ids(js, word: str) -> list[int]:
    """The word's single-token variants; raises if it has none."""
    ids = token_variants(js.lm.tokenizer, word)
    if not ids:
        raise ValueError(f"{word!r} has no single-token spelling; the J-lens cannot read it")
    return ids


def _project_word(js, word: str):
    ids = _word_ids(js, word)
    return ProjectOut(js, lens_vectors(js.lens, js.lm.unembedding_matrix, ids, js.band))


def _clamp_word(js, word: str):
    ids = _word_ids(js, word)
    caps = require_stats(js).caps(ids, js.band, js.cfg.clamp_z)
    return ReadoutClamp(js, lens_vectors(js.lens, js.lm.unembedding_matrix, ids, js.band), caps)


def _random_word_tokens(tokenizer, n: int, rng: random.Random) -> list[int]:
    """n random space-prefixed alphabetic tokens with at least 4 letters."""
    ids: list[int] = []
    while len(ids) < n:
        i = rng.randrange(len(tokenizer))
        text = tokenizer.decode([i])
        if text.startswith(" ") and text[1:].isalpha() and len(text) >= 5 and i not in ids:
            ids.append(i)
    return ids


@torch.no_grad()
def _clean_top_tokens_mask(js, enc: Encoded, k: int) -> torch.Tensor:
    """[batch, seq_len, vocab] bool: the clean pass's top-k output tokens at each position.
    TODO(debt): paper says "tokens that appear in the top-10 tokens of a clean
    forward pass" without saying per position or final position; we use per position."""
    logits = forward_logits(js.lm, enc.input_ids, enc.attention_mask).float()
    mask = torch.zeros_like(logits, dtype=torch.bool)
    return mask.scatter_(-1, logits.topk(k, dim=-1).indices, True)
