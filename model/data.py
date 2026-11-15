"""Batching for training, generic over the example type: a `to_text` function turns
one example into (prompt, completion) text, and examples ride along in the batch."""

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any, TypeVar

import torch

from model.inference import IGNORE_INDEX
from model.loading import LanguageModel

T = TypeVar("T")
ExampleText = Callable[[LanguageModel, T], tuple[str, str]]   # example -> (prompt, completion)


@dataclass
class Batch:
    input_ids: torch.Tensor       # [batch, seq], left-padded
    attention_mask: torch.Tensor  # [batch, seq]
    labels: torch.Tensor          # [batch, seq]; IGNORE_INDEX on prompt and padding
    examples: list[Any]           # row i of every tensor belongs to examples[i], passed through untouched


def example_ids(lm: LanguageModel, example: T, to_text: ExampleText) -> tuple[list[int], list[int]]:
    """(token ids, labels) for prompt + completion + EOS. The prompt is tokenised on its own,
    exactly as at inference, and only the completion and EOS are scored."""
    prompt, completion = to_text(lm, example)
    prompt_ids = lm.token_ids(prompt)
    completion_ids = lm.token_ids(completion + lm.tokenizer.eos_token)
    return prompt_ids + completion_ids, [IGNORE_INDEX] * len(prompt_ids) + completion_ids


def _left_pad(rows: list[list[int]], value: int) -> torch.Tensor:
    width = max(len(row) for row in rows)
    return torch.tensor([[value] * (width - len(row)) + row for row in rows])


def collate(lm: LanguageModel, examples: Sequence[T], to_text: ExampleText) -> Batch:
    """Tokenise and left-pad examples into a Batch on the model's device."""
    pairs = [example_ids(lm, ex, to_text) for ex in examples]
    ids = [p[0] for p in pairs]
    return Batch(
        input_ids=_left_pad(ids, lm.tokenizer.pad_token_id).to(lm.device),
        attention_mask=_left_pad([[1] * len(row) for row in ids], 0).to(lm.device),
        labels=_left_pad([p[1] for p in pairs], IGNORE_INDEX).to(lm.device),
        examples=list(examples),
    )
