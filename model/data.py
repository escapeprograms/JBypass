"""The Example shape shared with the dataset/eval code, and batching for training."""

from collections.abc import Callable, Sequence
from dataclasses import dataclass

import torch

from jbypass.model.inference import IGNORE_INDEX
from jbypass.model.loading import LanguageModel

PromptFormatter = Callable[[LanguageModel, str], str]   # e.g. prompts.chat_prompt


@dataclass(frozen=True)
class Example:
    question: str
    answer: str
    bridge: str | None = None   # the intermediate entity of a two-hop question, if known


@dataclass
class Batch:
    input_ids: torch.Tensor       # [batch, seq], left-padded
    attention_mask: torch.Tensor  # [batch, seq]
    labels: torch.Tensor          # [batch, seq]; IGNORE_INDEX on prompt and padding
    examples: list[Example]       # row i of every tensor belongs to examples[i]


def example_ids(lm: LanguageModel, example: Example, format_prompt: PromptFormatter) -> tuple[list[int], list[int]]:
    """(token ids, labels) for prompt + answer + EOS. The prompt is tokenised on its own,
    exactly as at inference, and only the answer and EOS are scored."""
    prompt_ids = lm.token_ids(format_prompt(lm, example.question))
    answer_ids = lm.token_ids(example.answer + lm.tokenizer.eos_token)
    return prompt_ids + answer_ids, [IGNORE_INDEX] * len(prompt_ids) + answer_ids


def _left_pad(rows: list[list[int]], value: int) -> torch.Tensor:
    width = max(len(row) for row in rows)
    return torch.tensor([[value] * (width - len(row)) + row for row in rows])


def collate(lm: LanguageModel, examples: Sequence[Example], format_prompt: PromptFormatter) -> Batch:
    """Tokenise and left-pad examples into a Batch on the model's device."""
    pairs = [example_ids(lm, ex, format_prompt) for ex in examples]
    ids = [p[0] for p in pairs]
    return Batch(
        input_ids=_left_pad(ids, lm.tokenizer.pad_token_id).to(lm.device),
        attention_mask=_left_pad([[1] * len(row) for row in ids], 0).to(lm.device),
        labels=_left_pad([p[1] for p in pairs], IGNORE_INDEX).to(lm.device),
        examples=list(examples),
    )
