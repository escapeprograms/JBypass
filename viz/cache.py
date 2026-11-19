"""Rendered results of the selected question, one per (prompt field, ablation, model), so showing a
model that already ran needs no load or generation. Plain Python only (no gradio, no torch)."""

from dataclasses import dataclass


@dataclass(frozen=True)
class RunKey:
    dataset: str
    name: str                       # item name
    field: str                      # prompt field
    condition: str                  # ablation
    model: str                      # model dropdown label

    @property
    def question(self) -> tuple[str, str]:
        """(dataset, item name): the scope the cache keeps results for."""
        return (self.dataset, self.name)


@dataclass(frozen=True)
class Shown:
    """What the viewer shows for one run, as strings (no tensors)."""
    summary: str                    # markdown summary, naming the model
    original: str                   # heatmap iframe HTML under the original (prefit) lens
    own: str                        # heatmap iframe HTML under the model's refit lens, or the note
                                    # saying how to fit one
    response: str                  # full generated text
    correct: bool                   # scored as eval scores it


class ResultCache:
    """Shown results of one question; a result for another question drops the old ones."""

    def __init__(self):
        self._entries: dict[RunKey, Shown] = {}

    def get(self, key: RunKey) -> Shown | None:
        """The stored result for `key`, or None (pure lookup)."""
        return self._entries.get(key)

    def put(self, key: RunKey, shown: Shown) -> None:
        """Store `shown`; if the cache holds another question, drop all of it first."""
        # TODO(debt): rendered pages are unbounded within one question; add a cap to viz.yaml if memory matters
        if any(k.question != key.question for k in self._entries):
            self._entries.clear()
        self._entries[key] = shown

    def entries(self, question: tuple[str, str]) -> list[tuple[RunKey, Shown]]:
        """Every stored (key, result) of `question`, in insertion order."""
        return [(k, s) for k, s in self._entries.items() if k.question == question]
