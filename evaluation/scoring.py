"""Scoring. Primary metric (proposal): exact match of a short greedy
continuation against the accepted answers, counting synonyms ("8" or "eight").
We also record the paper's metric (greedy next token == answer's first token);
on a 1.7B model it is harsh, because filler like " a" or a bare " " before a
digit often wins the first token.

The intervention is rebuilt for every decoding step from the current text, so
prompt-dependent ones (the paper's top-k ablation) see the right sequence. No
KV cache: each step is a full forward pass.
"""

import contextlib

import torch

FILLER_PREFIXES = ("a ", "an ", "the ", "called ", "number ")


@torch.no_grad()
def next_token_logits(js, prompt: str) -> torch.Tensor:
    """Final-position logits [vocab] under whatever interventions are active."""
    input_ids = js.model.encode(prompt)
    return js.hf_model(input_ids=input_ids).logits[0, -1].float()


def greedy_continuation(js, prompt: str, make_intervention, n_tokens: int) -> tuple[str, int]:
    """Greedy-decode n_tokens. make_intervention(text) -> context manager or None.
    Returns (continuation text, first generated token id)."""
    text, first = prompt, None
    for _ in range(n_tokens):
        with make_intervention(text) or contextlib.nullcontext():
            token = int(next_token_logits(js, text).argmax())
        first = token if first is None else first
        text += js.tokenizer.decode([token])
    return text[len(prompt):], first


def matches(continuation: str, answers) -> bool:
    """Exact match on the first line, after dropping filler like 'a' or 'the'."""
    t = continuation.strip().split("\n")[0].strip().lower().lstrip("\"'")
    for prefix in FILLER_PREFIXES:
        if t.startswith(prefix):
            t = t[len(prefix):]
    for a in (a.lower() for a in answers):
        if t.startswith(a) and (len(t) == len(a) or not t[len(a)].isalnum()):
            return True
    return False


def answer_token_ids(tokenizer, answers) -> set[int]:
    """First token of every spelling of every accepted answer."""
    ids = set()
    for a in answers:
        for text in (a, " " + a, a.lower(), " " + a.lower(), a.capitalize(), " " + a.capitalize()):
            ids.add(tokenizer(text, add_special_tokens=False).input_ids[0])
    # Qwen splits " 8" into " " + "8"; a bare-space first token is not an answer.
    return {i for i in ids if tokenizer.decode([i]).strip()}
