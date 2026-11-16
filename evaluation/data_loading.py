"""Load two-hop datasets into one item type. File access only; no model code."""

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class TwoHopItem:
    """One two-hop question.

    prompt: two-hop prompt, ends right before the answer.
    bridge: the intermediate entity (what we ablate).
    answers: accepted answers; scored on the greedy next token.
    swap_to: a different bridge of the same kind (wrong-bridge control / swap).
    swap_answers: answer if the bridge were swap_to (paper swap check), if known.
    hop1_prompt / hop2_prompt: one-hop components, if the dataset has them.
    """

    name: str
    category: str
    prompt: str
    bridge: str
    answers: tuple[str, ...]
    swap_to: str | None = None
    swap_answers: tuple[str, ...] = ()
    hop1_prompt: str | None = None
    hop2_prompt: str | None = None

    def __post_init__(self):
        # A trailing space becomes a lone space token in Qwen and derails the
        # model (the paper's prompts end with "is "); the answer follows anyway.
        for name in ("prompt", "hop1_prompt", "hop2_prompt"):
            value = getattr(self, name)
            if value is not None:
                object.__setattr__(self, name, value.rstrip())


def load_jbypass(path: Path) -> list[TwoHopItem]:
    """Our dataset (evaluation/data/build_jbypass_dataset.py)."""
    # TODO(debt): read_text() uses the platform encoding; the file is UTF-8 with accented
    # names, so pass encoding="utf-8" (here and in the other loaders) before running
    # anywhere Python is not in UTF-8 mode.
    rows = json.loads(Path(path).read_text())["items"]
    return [TwoHopItem(
        name=r["name"], category=r["category"], prompt=r["prompt"], bridge=r["bridge"],
        answers=tuple(r["answers"]), swap_to=r["swap_to"],
        hop1_prompt=r["hop1_prompt"], hop2_prompt=r["hop2_prompt"]) for r in rows]


def load_paper_probe_swap(path: Path) -> list[TwoHopItem]:
    """Anthropic's probe-swap.json: 90 two-hop prompts with bridge and swap target."""
    rows = json.loads(Path(path).read_text())["items"]
    return [TwoHopItem(
        name=r["name"], category=r["category"], prompt=r["prompt"],
        bridge=r["intermediate"], answers=(r["answer"],), swap_to=r["swap_to"],
        swap_answers=(r["swap_answer"],)) for r in rows]


def load_paper_multihop(path: Path) -> list[TwoHopItem]:
    """Anthropic's lens-eval-multihop.json (lens readout eval). Uses the first
    listed intermediate as the bridge."""
    rows = json.loads(Path(path).read_text())["items"]
    return [TwoHopItem(
        name=r["name"], category="multihop", prompt=r["prompt"],
        bridge=r["intermediates"][0], answers=(r["target"],)) for r in rows]


LOADERS = {
    "jbypass": load_jbypass,
    "paper_probe_swap": load_paper_probe_swap,
    "paper_multihop": load_paper_multihop,
}


def load_dataset(name: str, path: Path) -> list[TwoHopItem]:
    return LOADERS[name](path)
