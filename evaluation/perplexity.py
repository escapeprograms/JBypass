"""Sanity check from the proposal: perplexity on held-out text, to make sure
fine-tuning did not damage general abilities. Compare M against Mctrl / Mexp."""

import math

import torch


@torch.no_grad()
def perplexity(js, texts: list[str], max_len: int) -> float:
    """Token-weighted perplexity of js.hf_model over texts (truncated to max_len)."""
    total_nll, total_tokens = 0.0, 0
    for text in texts:
        ids = js.model.encode(text, max_length=max_len)
        if ids.shape[1] < 2:
            continue
        logits = js.hf_model(input_ids=ids).logits[0, :-1].float()
        nll = torch.nn.functional.cross_entropy(logits, ids[0, 1:], reduction="sum")
        total_nll += float(nll)
        total_tokens += ids.shape[1] - 1
    return math.exp(total_nll / total_tokens)
