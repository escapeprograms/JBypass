"""Sanity check from the proposal: perplexity on held-out text, to make sure
fine-tuning did not damage general abilities. Compare M against Mctrl / Mexp."""

import math

from model.inference import encode, sequence_nll
from model.loading import LanguageModel


def perplexity(lm: LanguageModel, texts: list[str], max_len: int, batch_size: int) -> float:
    """Token-weighted perplexity over all texts, each truncated to its first max_len
    tokens; texts under 2 tokens are skipped. Runs in left-padded batches."""
    usable = [t for t in texts if len(lm.token_ids(t)) >= 2]
    total_nll, total_tokens = 0.0, 0
    for start in range(0, len(usable), batch_size):
        nll, count = sequence_nll(lm, encode(lm, usable[start:start + batch_size], max_length=max_len))
        total_nll += float(nll.sum())
        total_tokens += int(count.sum())
    return math.exp(total_nll / total_tokens)
