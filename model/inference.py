"""Forward-only use: tokenising, generation, next-token logits and loss on text.

All public functions except `forward_logits` and `nll_per_row` run under no_grad.
Those two are the in-graph building blocks that training reuses.
"""

from collections.abc import Callable, Sequence
from dataclasses import dataclass

import torch
import torch.nn.functional as F

from model.config import GenerationSettings
from model.hooks import Intervention, apply_interventions
from model.loading import LanguageModel

IGNORE_INDEX = -100   # label value excluded from the loss (prompt and padding)


@dataclass(frozen=True)
class Encoded:
    input_ids: torch.Tensor       # [batch, seq], left-padded
    attention_mask: torch.Tensor  # [batch, seq], 0 on padding


InterventionsFor = Callable[[Encoded], Sequence[Intervention]]   # built from the current sequence


def encode(lm: LanguageModel, texts: Sequence[str], max_length: int | None = None) -> Encoded:
    """Tokenise without added special tokens, left-padded, on the model's device.
    With `max_length`, each text keeps only its first `max_length` tokens."""
    batch = lm.tokenizer(
        list(texts), return_tensors="pt", padding=True, add_special_tokens=False,
        truncation=max_length is not None, max_length=max_length,
    )
    return Encoded(batch.input_ids.to(lm.device), batch.attention_mask.to(lm.device))


def position_ids(attention_mask: torch.Tensor) -> torch.Tensor:
    """Positions counted from each row's first real token, so left padding does not shift them."""
    return (attention_mask.cumsum(-1) - 1).clamp(min=0)


def forward_logits(
    lm: LanguageModel,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    interventions: Sequence[Intervention] = (),
) -> torch.Tensor:
    """In-graph [batch, seq, vocab] logits with the interventions applied."""
    with apply_interventions(lm, interventions, attention_mask):
        out = lm.hf_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids(attention_mask),
            use_cache=False,
        )
    return out.logits


def nll_per_row(logits: torch.Tensor, labels: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Summed next-token NLL and the number of scored tokens per row. Labels equal to
    IGNORE_INDEX are skipped. Computed in fp32."""
    shift_logits = logits[:, :-1].float()
    shift_labels = labels[:, 1:]
    nll = F.cross_entropy(
        shift_logits.transpose(1, 2), shift_labels, ignore_index=IGNORE_INDEX, reduction="none"
    )
    return nll.sum(-1), (shift_labels != IGNORE_INDEX).sum(-1)


@torch.no_grad()
def next_token_logits(
    lm: LanguageModel, prompts: Sequence[str], interventions: Sequence[Intervention] = ()
) -> torch.Tensor:
    """[batch, vocab] logits for the token after each prompt."""
    enc = encode(lm, prompts)
    return forward_logits(lm, enc.input_ids, enc.attention_mask, interventions)[:, -1]


@torch.no_grad()
def sequence_nll(
    lm: LanguageModel, enc: Encoded, interventions: Sequence[Intervention] = ()
) -> tuple[torch.Tensor, torch.Tensor]:
    """Summed next-token NLL and scored-token count per row, over each row's real tokens
    after the first. A row with fewer than two real tokens scores nothing."""
    # Padding and each row's first real token both have position 0; neither is a target.
    labels = enc.input_ids.masked_fill(position_ids(enc.attention_mask) == 0, IGNORE_INDEX)
    logits = forward_logits(lm, enc.input_ids, enc.attention_mask, interventions)
    return nll_per_row(logits, labels)


@torch.no_grad()
def sequence_loss(
    lm: LanguageModel, texts: Sequence[str], interventions: Sequence[Intervention] = ()
) -> torch.Tensor:
    """[batch] mean next-token NLL over each text's real tokens after the first
    (exp() gives perplexity). A text needs at least two tokens."""
    total, count = sequence_nll(lm, encode(lm, texts), interventions)
    return total / count


@torch.no_grad()
def greedy_tokens(
    lm: LanguageModel, enc: Encoded, max_new_tokens: int, interventions_for: InterventionsFor
) -> torch.Tensor:
    """[batch, n_new] greedy tokens; stops early once every row has produced EOS.
    `interventions_for` is called before every step with the sequence grown so far,
    so sequence-dependent edits see the tokens they act on."""
    # TODO(debt): no KV cache — each new token reruns the full sequence so hooks always see
    # the whole sequence. Add caching behind a cached-equals-uncached test when long
    # generations are needed.
    ids, mask = enc.input_ids, enc.attention_mask
    done = torch.zeros(ids.shape[0], dtype=torch.bool, device=ids.device)
    for _ in range(max_new_tokens):
        interventions = interventions_for(Encoded(ids, mask))
        next_ids = forward_logits(lm, ids, mask, interventions)[:, -1].argmax(-1)
        ids = torch.cat([ids, next_ids[:, None]], dim=1)
        mask = torch.cat([mask, torch.ones_like(next_ids[:, None])], dim=1)
        done |= next_ids == lm.tokenizer.eos_token_id
        if done.all():
            break
    return ids[:, enc.input_ids.shape[1]:]


def _decode_until_eos(lm: LanguageModel, tokens: torch.Tensor) -> list[str]:
    """Decode each row up to (not including) its first EOS."""
    texts = []
    for row in tokens.tolist():
        end = row.index(lm.tokenizer.eos_token_id) if lm.tokenizer.eos_token_id in row else len(row)
        texts.append(lm.tokenizer.decode(row[:end], skip_special_tokens=True))
    return texts


@torch.no_grad()
def generate(
    lm: LanguageModel,
    prompts: Sequence[str],
    settings: GenerationSettings,
    interventions: Sequence[Intervention] = (),
) -> list[str]:
    """Greedy continuation of each prompt: new text only, stopped at EOS."""
    tokens = greedy_tokens(lm, encode(lm, prompts), settings.max_new_tokens, lambda enc: interventions)
    return _decode_until_eos(lm, tokens)
