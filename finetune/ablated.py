"""Training under an eval condition: the text of a two-hop item, and an intervention
factory that applies the condition's builder to every row of a training batch."""

from evaluation.conditions import CONDITIONS, item_rng
from evaluation.data_loading import TwoHopItem
from jspace import JSpace
from model.data import Batch
from model.hooks import PerRow
from model.inference import Encoded
from model.loading import LanguageModel
from model.training import InterventionFactory


def item_text(lm: LanguageModel, item: TwoHopItem) -> tuple[str, str]:
    """(prompt, completion): the raw `Fact: ...` prompt the lens was fit on and eval
    scores, and the first accepted answer after a space."""
    return item.prompt, " " + item.answers[0]


def condition_factory(js: JSpace, condition: str, seed) -> InterventionFactory:
    """Per batch: the condition's interventions for each row's item, each row seeded by
    item_rng(seed, item, condition) as in eval, wrapped in one PerRow."""
    build = CONDITIONS[condition]

    def factory(batch: Batch):
        rows = [build(js, item, _row(batch, i), item_rng(seed, item, condition))
                for i, item in enumerate(batch.examples)]
        return [PerRow(rows)]
    return factory


def _row(batch: Batch, i: int) -> Encoded:
    """Row i of the batch (left padding included) as a one-row Encoded."""
    return Encoded(batch.input_ids[i:i + 1], batch.attention_mask[i:i + 1])
