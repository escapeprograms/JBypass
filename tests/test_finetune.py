"""Training under an eval condition (finetune/) on tiny models with a synthetic lens, CPU."""

import math

import torch

from evaluation.conditions import CONDITIONS
from evaluation.data_loading import TwoHopItem
from finetune.ablated import condition_factory, item_text
from finetune.config import TRAIN_CONDITIONS
from jspace import ReadoutClamp
from jspace.interventions import ProjectOut
from model.config import TrainSettings
from model.hooks import PerRow
from model.lora import attach_lora
from model.training import train

from conftest import LORA, synthetic_js, synthetic_stats

ITEMS = [
    TwoHopItem(name="a-0", category="t", prompt="Fact: The capital of the country of the Eiffel Tower is",
               bridge="France", answers=("Paris",)),
    TwoHopItem(name="b-0", category="t", prompt="Fact: The language spoken in the country of the Colosseum is",
               bridge="Italy", answers=("Italian",)),
    TwoHopItem(name="c-0", category="t", prompt="Fact: The continent of the country of Mount Fuji is",
               bridge="Japan", answers=("Asia",)),
]
STEPS = 30


def test_item_text_is_raw_prompt_and_spaced_answer(qwen_lm):
    assert item_text(qwen_lm, ITEMS[0]) == (ITEMS[0].prompt, " Paris")


def test_training_conditions_are_eval_conditions():
    assert set(TRAIN_CONDITIONS) <= set(CONDITIONS) and "bridge" in TRAIN_CONDITIONS


def test_condition_factory_builds_one_per_row_ablation_per_batch(lm):
    js = synthetic_js(lm)
    calls, losses, before = _train_checking_rows(lm, js, "bridge", ProjectOut)
    assert len(calls) == STEPS
    assert losses[-1] < 0.5 * losses[0], losses
    changed = {n for n, p in lm.hf_model.named_parameters() if not torch.equal(p, before[n])}
    assert changed and all("lora_" in n for n in changed)


def test_condition_factory_bridge_clamp_trains_under_readout_clamps(lm):
    js = synthetic_js(lm)
    js.readout_stats = synthetic_stats(js)
    calls, losses, _ = _train_checking_rows(lm, js, "bridge_clamp", ReadoutClamp)
    assert len(calls) == STEPS and all(math.isfinite(l) for l in losses)


def _train_checking_rows(lm, js, condition: str, kind: type):
    """Train STEPS steps under `condition`, asserting each batch gets one PerRow whose rows
    are single `kind` interventions on the band. Returns (batches seen, losses, parameters
    before training)."""
    calls = []
    inner = condition_factory(js, condition, seed=0)

    def factory(batch):
        calls.append(batch)
        interventions = inner(batch)
        assert len(interventions) == 1 and isinstance(interventions[0], PerRow)
        rows = interventions[0].rows
        assert len(rows) == len(batch.examples)
        assert all(len(r) == 1 and isinstance(r[0], kind) and r[0].layers == js.band for r in rows)
        return interventions

    with torch.no_grad():
        lm.unembedding_matrix.mul_(20)   # init logits are capped near ln(vocab); see test_loss_falls_on_toy_set
    attach_lora(lm, LORA)
    before = {n: p.detach().clone() for n, p in lm.hf_model.named_parameters()}
    losses = train(lm, ITEMS, TrainSettings(1e-2, batch_size=3, steps=STEPS, seed=0), item_text, factory)
    return calls, losses, before
