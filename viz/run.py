"""One generation under one condition, kept as plain Python values (never pins GPU memory)."""

from dataclasses import dataclass

from evaluation.conditions import interventions_for_item
from evaluation.data_loading import TwoHopItem
from evaluation.runner import field_answers
from evaluation.scoring import matches
from model.inference import encode, greedy_tokens


@dataclass(frozen=True)
class RunResult:
    item: TwoHopItem
    field: str                      # prompt field used, e.g. "prompt"
    condition: str                  # name in evaluation.conditions.CONDITIONS
    seed: int                       # seed of the condition's item_rng
    prompt: str
    prompt_ids: tuple[int, ...]     # model.inference.encode of the prompt
    response_ids: tuple[int, ...]   # greedy tokens (may end with EOS)
    response_text: str              # decoded as evaluation.scoring.greedy_continuation does
    answers: tuple[str, ...]        # accepted answers for this field
    correct: bool                   # evaluation.scoring.matches(response_text, answers)


def run_example(js, item: TwoHopItem, field: str, condition: str, seed: int,
                max_new_tokens: int) -> RunResult:
    """Greedy-decode `item`'s `field` prompt under `condition`, scored as eval scores it."""
    prompt = getattr(item, field)
    enc = encode(js.lm, [prompt])
    interventions_for = interventions_for_item(js, item, condition, seed)
    response = greedy_tokens(js.lm, enc, max_new_tokens, interventions_for)[0].tolist()
    text = js.lm.tokenizer.decode(response)
    answers = tuple(field_answers(item, field))
    return RunResult(
        item=item, field=field, condition=condition, seed=seed, prompt=prompt,
        prompt_ids=tuple(enc.input_ids[0].tolist()), response_ids=tuple(response),
        response_text=text, answers=answers, correct=matches(text, answers),
    )
