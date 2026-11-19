"""Run items x prompt types x ablation conditions and summarize accuracy."""

from collections import defaultdict

from evaluation.conditions import interventions_for_item
from evaluation.scoring import answer_token_ids, greedy_continuation, matches
from jspace import token_variants


def field_answers(item, field: str) -> tuple[str, ...]:
    """hop1's answer is the bridge itself; every other prompt is answered by item.answers."""
    return (item.bridge,) if field == "hop1_prompt" else item.answers


def readable_items(js, items):
    """Drop items whose bridge (or swap_to) has no single-token spelling."""
    return tokenizer_readable_items(js.lm.tokenizer, items)


def tokenizer_readable_items(tokenizer, items):
    """readable_items with only a tokenizer (no model needed)."""
    def ok(word):
        return word is None or bool(token_variants(tokenizer, word))
    return [it for it in items if ok(it.bridge) and ok(it.swap_to)]


def score_one(js, item, field: str, condition: str, seed: int, n_tokens: int) -> dict:
    """correct: exact match of the greedy continuation (primary metric).
    next_token_correct: the paper's stricter first-token metric."""
    prompt = getattr(item, field)
    answers = field_answers(item, field)
    interventions_for = interventions_for_item(js, item, condition, seed)
    continuation, first = greedy_continuation(js.lm, prompt, interventions_for, n_tokens)
    return {"name": item.name, "field": field, "condition": condition,
            "correct": matches(continuation, answers),
            "next_token_correct": first in answer_token_ids(js.lm.tokenizer, answers),
            "output": continuation}


def evaluate(js, items, conditions, fields, seed: int = 0, n_tokens: int = 5) -> list[dict]:
    """One record per (item, field, condition). Fields an item lacks are skipped."""
    records = []
    for item in items:
        for field in fields:
            if getattr(item, field) is None:
                continue
            for condition in conditions:
                records.append(score_one(js, item, field, condition, seed, n_tokens))
    return records


def solved_items(js, items, fields, n_tokens: int = 5) -> list:
    """Items the clean model answers correctly on every available field."""
    records = evaluate(js, items, ["none"], fields, n_tokens=n_tokens)
    failed = {r["name"] for r in records if not r["correct"]}
    return [it for it in items if it.name not in failed]


def accuracy_table(records, key: str = "correct") -> dict:
    """{field: {condition: (accuracy, n)}}. key: 'correct' or 'next_token_correct'."""
    hits = defaultdict(lambda: defaultdict(list))
    for r in records:
        hits[r["field"]][r["condition"]].append(r[key])
    return {f: {c: (sum(v) / len(v), len(v)) for c, v in by_c.items()} for f, by_c in hits.items()}


def format_table(table: dict, conditions) -> str:
    lines = [f"{'prompt type':14s}" + "".join(f"{c:>14s}" for c in conditions)]
    for field, by_c in table.items():
        cells = [f"{by_c[c][0]:14.2f}" if c in by_c else f"{'-':>14s}" for c in conditions]
        lines.append(f"{field:14s}" + "".join(cells))
    return "\n".join(lines)
