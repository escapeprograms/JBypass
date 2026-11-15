"""Next-token scoring, as in the paper ("greedy next-token == answer")."""

import torch


@torch.no_grad()
def next_token_logits(js, prompt: str) -> torch.Tensor:
    """Final-position logits [vocab] under whatever interventions are active."""
    input_ids = js.model.encode(prompt)
    return js.hf_model(input_ids=input_ids).logits[0, -1].float()


def answer_token_ids(tokenizer, answers) -> set[int]:
    """First token of every spelling of every accepted answer. Covers prompts
    that end with or without a trailing space."""
    ids = set()
    for a in answers:
        for text in (a, " " + a, a.lower(), " " + a.lower(), a.capitalize(), " " + a.capitalize()):
            ids.add(tokenizer(text, add_special_tokens=False).input_ids[0])
    # Qwen splits " 8" into " " + "8"; a bare-space first token is not an answer.
    return {i for i in ids if tokenizer.decode([i]).strip()}


def is_correct(logits: torch.Tensor, answer_ids: set[int]) -> bool:
    return int(logits.argmax()) in answer_ids
