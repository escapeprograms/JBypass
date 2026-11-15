"""Train / test splits by bridge entity, so test_unseen bridges never appear in training.

train        questions for (1 - unseen_frac) of the bridges, minus held-out ones
test_seen    one held-out question per seen bridge that has 2+ questions
test_unseen  every question for the remaining bridges
"""

import random
from collections import defaultdict


def split_by_bridge(items, unseen_frac: float, seed: int) -> dict[str, list[str]]:
    """Returns {split: [item names]}."""
    rng = random.Random(seed)
    by_bridge = defaultdict(list)
    for it in items:
        by_bridge[it.bridge].append(it.name)
    bridges = sorted(by_bridge)
    rng.shuffle(bridges)
    unseen = set(bridges[: round(unseen_frac * len(bridges))])
    splits = {"train": [], "test_seen": [], "test_unseen": []}
    for bridge in sorted(by_bridge):
        names = sorted(by_bridge[bridge])
        if bridge in unseen:
            splits["test_unseen"] += names
            continue
        rng.shuffle(names)
        held = names[:1] if len(names) >= 2 else []
        splits["test_seen"] += held
        splits["train"] += names[len(held):]
    return splits
