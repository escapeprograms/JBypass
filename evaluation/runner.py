"""Run items x prompt types x ablation conditions and summarize accuracy."""

import random
from collections import defaultdict

from evaluation.conditions import CONDITIONS
from evaluation.scoring import answer_token_ids, is_correct, next_token_logits
from jspace import token_variants


def field_answers(item, field: str) -> tuple[str, ...]:
    """hop1's answer is the bridge itself; every other prompt is answered by item.answers."""
    return (item.bridge,) if field == "hop1_prompt" else item.answers


def readable_items(js, items):
    """Drop items whose bridge (or swap_to) has no single-token spelling."""
    def ok(word):
        return word is None or bool(token_variants(js.tokenizer, word))
    return [it for it in items if ok(it.bridge) and ok(it.swap_to)]


def score_one(js, item, field: str, condition: str, rng) -> dict:
    prompt = getattr(item, field)
    intervention = CONDITIONS[condition](js, item, prompt, rng)
    if intervention is None:
        logits = next_token_logits(js, prompt)
    else:
        with intervention:
            logits = next_token_logits(js, prompt)
    correct = is_correct(logits, answer_token_ids(js.tokenizer, field_answers(item, field)))
    pred = js.tokenizer.decode([int(logits.argmax())])
    return {"name": item.name, "field": field, "condition": condition,
            "correct": correct, "pred": pred}


def evaluate(js, items, conditions, fields, seed: int = 0) -> list[dict]:
    """One record per (item, field, condition). Fields an item lacks are skipped."""
    records = []
    for item in items:
        for field in fields:
            if getattr(item, field) is None:
                continue
            for condition in conditions:
                rng = random.Random(f"{seed}-{item.name}-{condition}")
                records.append(score_one(js, item, field, condition, rng))
    return records


def solved_items(js, items, fields) -> list:
    """Items the clean model answers correctly on every available field."""
    records = evaluate(js, items, ["none"], fields)
    failed = {r["name"] for r in records if not r["correct"]}
    return [it for it in items if it.name not in failed]


def accuracy_table(records) -> dict:
    """{field: {condition: (accuracy, n)}}"""
    hits = defaultdict(lambda: defaultdict(list))
    for r in records:
        hits[r["field"]][r["condition"]].append(r["correct"])
    return {f: {c: (sum(v) / len(v), len(v)) for c, v in by_c.items()} for f, by_c in hits.items()}


def format_table(table: dict, conditions) -> str:
    lines = [f"{'prompt type':14s}" + "".join(f"{c:>14s}" for c in conditions)]
    for field, by_c in table.items():
        cells = [f"{by_c[c][0]:14.2f}" if c in by_c else f"{'-':>14s}" for c in conditions]
        lines.append(f"{field:14s}" + "".join(cells))
    return "\n".join(lines)
