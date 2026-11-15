"""Scoring. Primary metric (proposal): exact match of a short greedy
continuation against the accepted answers, counting synonyms ("8" or "eight").
We also record the paper's metric (greedy next token == answer's first token);
on a 1.7B model it is harsh, because filler like " a" or a bare " " before a
digit often wins the first token.

Decoding is `model.inference.greedy_tokens`: the interventions are rebuilt for every
step from the current token ids, so sequence-dependent ones (the paper's top-k
ablation) see the right sequence. No KV cache: each step is a full forward pass.
"""

from model.inference import InterventionsFor, encode, greedy_tokens
from model.loading import LanguageModel

FILLER_PREFIXES = ("a ", "an ", "the ", "called ", "number ")


def greedy_continuation(lm: LanguageModel, prompt: str, interventions_for: InterventionsFor,
                        n_tokens: int) -> tuple[str, int]:
    """Greedy-decode up to n_tokens (stops early at EOS). interventions_for(enc) gives
    the interventions for each step. Returns (continuation text, first generated token id)."""
    tokens = greedy_tokens(lm, encode(lm, [prompt]), n_tokens, interventions_for)[0].tolist()
    return lm.tokenizer.decode(tokens), tokens[0]


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
