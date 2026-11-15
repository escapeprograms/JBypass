"""Ablation conditions. Each builder takes (js, item, prompt, rng) and returns a
jspace intervention (a context manager) or None for the clean run. Add a new
condition by writing a builder and registering it in CONDITIONS.

  none          clean model
  bridge        project out the bridge's J-lens vectors over the band   (the intervention)
  wrong_bridge  same for item.swap_to, another bridge of the same kind  (specificity control)
  random_tok    same number of J-lens vectors of random word tokens     (Mrand-style control)
  random_dir    same number of random Gaussian directions               (any-ablation control)
  topk          paper's ablation: top-k active J-lens vectors per position, excluding
                the clean pass's top-k output tokens                    (positive control)
"""

import random

import torch

from jspace import ProjectOut, TopKAblation, lens_vectors, token_variants


def build_none(js, item, prompt, rng):
    return None


def build_bridge(js, item, prompt, rng):
    return _project_word(js, item.bridge)


def build_wrong_bridge(js, item, prompt, rng):
    return _project_word(js, item.swap_to)


def build_random_tok(js, item, prompt, rng):
    n = len(token_variants(js.tokenizer, item.bridge))
    ids = _random_word_tokens(js.tokenizer, n, rng)
    return ProjectOut(js, lens_vectors(js.lens, js.W_U, ids, js.band))


def build_random_dir(js, item, prompt, rng):
    n = len(token_variants(js.tokenizer, item.bridge))
    gen = torch.Generator().manual_seed(rng.randrange(2**31))
    d = js.model.d_model
    return ProjectOut(js, {l: torch.randn(n, d, generator=gen) for l in js.band})


def build_topk(js, item, prompt, rng):
    exclude = _clean_top_tokens_mask(js, prompt, js.cfg.ablation_exclude_top_k)
    return TopKAblation(js, js.band, js.cfg.ablation_top_k, exclude)


CONDITIONS = {
    "none": build_none,
    "bridge": build_bridge,
    "wrong_bridge": build_wrong_bridge,
    "random_tok": build_random_tok,
    "random_dir": build_random_dir,
    "topk": build_topk,
}


def _project_word(js, word: str):
    ids = token_variants(js.tokenizer, word)
    if not ids:
        raise ValueError(f"{word!r} has no single-token spelling; the J-lens cannot read it")
    return ProjectOut(js, lens_vectors(js.lens, js.W_U, ids, js.band))


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
def _clean_top_tokens_mask(js, prompt: str, k: int) -> torch.Tensor:
    """[1, seq_len, vocab] bool: the clean pass's top-k output tokens at each position.
    TODO(debt): paper says "tokens that appear in the top-10 tokens of a clean
    forward pass" without saying per position or final position; we use per position."""
    input_ids = js.model.encode(prompt)
    logits = js.hf_model(input_ids=input_ids).logits.float()
    mask = torch.zeros_like(logits, dtype=torch.bool)
    return mask.scatter_(-1, logits.topk(k, dim=-1).indices, True)
