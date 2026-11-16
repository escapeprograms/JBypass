import pytest
import torch

from model.config import TrainSettings
from model.data import collate
from model.hooks import capture_residuals
from model.inference import IGNORE_INDEX, encode, next_token_logits
from model.lora import adapter_disabled, attach_lora, load_adapter, save_adapter, saved_profile
from model.prompts import chat_prompt
from model.training import example_batches, train

from conftest import LORA, TRAIN, build_gpt2, build_qwen
from dummies import Identity, ZeroOut

EXAMPLES = [   # (question, answer); train is generic over the example type
    ("What colour is the sky?", "Blue"),
    ("How many legs does a spider have?", "Eight"),
    ("Capital of France?", "Paris"),
]


def chat_text(lm, example):
    """to_text for the toy examples: (chat-formatted question, answer)."""
    return chat_prompt(lm, example[0]), example[1]
PROMPT = ["Question: what is two plus two?"]


def perturb_adapter(lm):
    """LoRA B starts at zero (adapter is a no-op); give it values so it changes outputs."""
    generator = torch.Generator().manual_seed(3)
    with torch.no_grad():
        for name, param in lm.hf_model.named_parameters():
            if "lora_B" in name:
                param.copy_(torch.randn(param.shape, generator=generator) * 0.5)


# --- data ---------------------------------------------------------------------

def test_collate_text_is_prompt_plus_completion_plus_eos(qwen_lm):
    batch = collate(qwen_lm, EXAMPLES[:2], chat_text)
    row = batch.input_ids[1][batch.attention_mask[1].bool()].tolist()
    expected = chat_prompt(qwen_lm, EXAMPLES[1][0]) + "Eight" + qwen_lm.tokenizer.eos_token
    assert qwen_lm.tokenizer.decode(row) == expected


def test_collate_labels_mask_prompt_and_padding(qwen_lm):
    batch = collate(qwen_lm, EXAMPLES, chat_text)
    assert batch.examples == EXAMPLES
    for i, ex in enumerate(EXAMPLES):
        scored = batch.labels[i][batch.labels[i] != IGNORE_INDEX].tolist()
        assert scored == qwen_lm.token_ids(ex[1] + qwen_lm.tokenizer.eos_token)
    assert (batch.labels[batch.attention_mask == 0] == IGNORE_INDEX).all()
    assert batch.attention_mask[:, -1].all()   # left-padded


def test_example_batches_cycle_and_cover(qwen_lm):
    batches = list(example_batches(EXAMPLES, TrainSettings(1e-3, batch_size=2, steps=3, seed=0)))
    assert [len(b) for b in batches] == [2, 2, 2]
    assert set(sum(batches, [])) == set(EXAMPLES)


# --- lora ---------------------------------------------------------------------

def test_hooks_still_fire_after_attaching_lora(lm):
    blocks_before = list(lm.blocks)
    attach_lora(lm, LORA)
    assert list(lm.blocks) == blocks_before
    base = next_token_logits(lm, PROMPT)
    assert not torch.allclose(next_token_logits(lm, PROMPT, [ZeroOut(layers=[1])]), base)
    enc = encode(lm, PROMPT)
    with capture_residuals(lm, [lm.n_layers - 1]) as resids:
        logits = lm.hf_model(input_ids=enc.input_ids, attention_mask=enc.attention_mask).logits
    torch.testing.assert_close(lm.unembed(resids[lm.n_layers - 1]), logits)


def test_lora_weights_are_fp32_over_bf16_base(tokenizer):
    lm = build_qwen(tokenizer)
    lm.hf_model.to(torch.bfloat16)
    attach_lora(lm, LORA)
    trainable = {n: p.dtype for n, p in lm.hf_model.named_parameters() if p.requires_grad}
    assert trainable and all("lora_" in n for n in trainable)
    assert set(trainable.values()) == {torch.float32}
    assert lm.dtype == torch.bfloat16
    losses = train(lm, EXAMPLES, TRAIN, chat_text)
    assert all(torch.isfinite(torch.tensor(losses)))


def test_adapter_disabled_reproduces_base_logits(lm):
    base = next_token_logits(lm, PROMPT)
    attach_lora(lm, LORA)
    perturb_adapter(lm)
    assert not torch.allclose(next_token_logits(lm, PROMPT), base)
    with adapter_disabled(lm):
        torch.testing.assert_close(next_token_logits(lm, PROMPT), base)
    assert not torch.allclose(next_token_logits(lm, PROMPT), base)


def test_save_load_round_trip(tokenizer, tmp_path):
    lm = build_qwen(tokenizer)
    attach_lora(lm, LORA)
    perturb_adapter(lm)
    tuned = next_token_logits(lm, PROMPT)
    save_adapter(lm, tmp_path)
    fresh = build_qwen(tokenizer)
    load_adapter(fresh, tmp_path)
    torch.testing.assert_close(next_token_logits(fresh, PROMPT), tuned)


def test_saved_profile_reads_the_saved_name(tokenizer, tmp_path):
    lm = build_qwen(tokenizer)
    attach_lora(lm, LORA)
    save_adapter(lm, tmp_path)
    assert saved_profile(tmp_path) == saved_profile(str(tmp_path)) == lm.profile.name


def test_loading_adapter_onto_different_profile_raises(tokenizer, tmp_path):
    lm = build_qwen(tokenizer)
    attach_lora(lm, LORA)
    save_adapter(lm, tmp_path)
    with pytest.raises(ValueError):
        load_adapter(build_gpt2(tokenizer), tmp_path)


# --- training -----------------------------------------------------------------

def test_only_adapter_weights_change(lm):
    attach_lora(lm, LORA)
    before = {n: p.detach().clone() for n, p in lm.hf_model.named_parameters()}
    train(lm, EXAMPLES, TRAIN, chat_text)
    changed = {n for n, p in lm.hf_model.named_parameters() if not torch.equal(p, before[n])}
    assert changed and all("lora_" in n for n in changed)
    assert all(p.grad is None for p in lm.hf_model.parameters())
    assert not lm.hf_model.training


def test_loss_falls_on_toy_set(lm):
    # At init the unembedding rows are tiny (std 0.02), which caps logits near ln(vocab)
    # no matter what LoRA does; scale them so the toy set can be fitted.
    with torch.no_grad():
        lm.unembedding_matrix.mul_(20)
    attach_lora(lm, LORA)
    losses = train(lm, EXAMPLES, TrainSettings(1e-2, batch_size=3, steps=30, seed=0), chat_text)
    assert losses[-1] < 0.5 * losses[0], losses


def test_intervention_factory_called_once_per_batch_with_matching_rows(lm):
    attach_lora(lm, LORA)
    calls, steps = [], []
    def factory(batch):
        calls.append(batch)
        return [Identity(layers=[1])]
    train(lm, EXAMPLES, TRAIN, chat_text, factory, lambda step, loss, m: steps.append(step))
    assert len(calls) == TRAIN.steps and steps == list(range(TRAIN.steps))
    assert all(b.input_ids.shape[0] == len(b.examples) == TRAIN.batch_size for b in calls)


def _adapter_params_by_block(lm):
    return [[p for n, p in block.named_parameters() if "lora_" in n] for block in lm.blocks]


def test_interventions_sit_in_the_training_graph(lm):
    """Zeroing block 1's output cuts the path from adapters in blocks 0 and 1 to the loss,
    while adapters after it still train: the intervention is part of the graph."""
    attach_lora(lm, LORA)
    perturb_adapter(lm)
    before = [[p.detach().clone() for p in ps] for ps in _adapter_params_by_block(lm)]
    train(lm, EXAMPLES, TRAIN, chat_text, lambda batch: [ZeroOut(layers=[1])])
    moved = [any(not torch.equal(p, b) for p, b in zip(ps, bs))
             for ps, bs in zip(_adapter_params_by_block(lm), before)]
    assert moved == [False, False, True]
