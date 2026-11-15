"""The optimisation loop. Independent of LoRA (it trains whatever has requires_grad)
and of J-space (interventions come from a factory)."""

from collections.abc import Callable, Iterator, Sequence

import torch

from model.config import TrainSettings
from model.data import Batch, ExampleText, T, collate
from model.hooks import Intervention
from model.inference import forward_logits, nll_per_row
from model.loading import LanguageModel

InterventionFactory = Callable[[Batch], Sequence[Intervention]]
StepCallback = Callable[[int, float, LanguageModel], None]   # step, loss, model


def no_interventions(batch: Batch) -> Sequence[Intervention]:
    """Default factory: train without interventions (the no-ablation control)."""
    return ()


def ignore_step(step: int, loss: float, lm: LanguageModel) -> None:
    """Default callback: do nothing."""


def example_batches(examples: Sequence[T], settings: TrainSettings) -> Iterator[list[T]]:
    """`settings.steps` batches, cycling through the examples in a fresh seeded shuffle each pass."""
    generator = torch.Generator().manual_seed(settings.seed)
    order: list[int] = []
    for _ in range(settings.steps):
        while len(order) < settings.batch_size:
            order += torch.randperm(len(examples), generator=generator).tolist()
        yield [examples[i] for i in order[: settings.batch_size]]
        order = order[settings.batch_size:]


def batch_loss(lm: LanguageModel, batch: Batch, interventions: Sequence[Intervention]) -> torch.Tensor:
    """Mean NLL over the answer tokens of the batch, in-graph, computed in fp32."""
    logits = forward_logits(lm, batch.input_ids, batch.attention_mask, interventions)
    total, count = nll_per_row(logits, batch.labels)
    return total.sum() / count.sum()


def train(
    lm: LanguageModel,
    examples: Sequence[T],
    settings: TrainSettings,
    to_text: ExampleText,
    intervention_factory: InterventionFactory = no_interventions,
    on_step: StepCallback = ignore_step,
) -> list[float]:
    """AdamW on every parameter with requires_grad. `to_text(lm, example)` gives the
    (prompt, completion) text; examples are passed to the batch untouched. The factory is
    called once per batch; row i of the residual belongs to batch.examples[i]. Returns
    the loss per step and leaves the model in eval mode."""
    params = [p for p in lm.hf_model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=settings.learning_rate)
    lm.hf_model.train()
    losses = []
    for step, chunk in enumerate(example_batches(examples, settings)):
        batch = collate(lm, chunk, to_text)
        optimizer.zero_grad(set_to_none=True)
        loss = batch_loss(lm, batch, intervention_factory(batch))
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
        on_step(step, losses[-1], lm)
    optimizer.zero_grad(set_to_none=True)
    lm.hf_model.eval()
    return losses
