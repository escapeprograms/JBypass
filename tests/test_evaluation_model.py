"""evaluation on the shared model: decoding, condition builders and perplexity
(tiny Qwen-shaped model, synthetic lens, CPU)."""

import math

import pytest
import torch

from evaluation.conditions import CONDITIONS, interventions_for_item, item_rng
from evaluation.data_loading import TwoHopItem
from evaluation.perplexity import perplexity
from evaluation.run_readout_stats import reference_texts
from evaluation.scoring import answer_index, greedy_continuation, is_answer_token
from jspace import ReadoutClamp, token_variants
from model.inference import encode, forward_logits, greedy_tokens

from conftest import synthetic_js, synthetic_stats

ITEM = TwoHopItem(name="x-0", category="test", prompt="Fact: The capital of the country is",
                  bridge="France", answers=("Paris",), swap_to="Spain")


def test_greedy_continuation_returns_text_and_first_token(qwen_lm):
    text, first = greedy_continuation(qwen_lm, ITEM.prompt, lambda enc: (), 3)
    tokens = greedy_tokens(qwen_lm, encode(qwen_lm, [ITEM.prompt]), 3, lambda enc: ())[0].tolist()
    assert text == qwen_lm.tokenizer.decode(tokens) and first == tokens[0]


def test_builders_return_sequences_and_none_is_empty(qwen_lm):
    js = synthetic_js(qwen_lm)
    js.readout_stats = synthetic_stats(js)
    enc = encode(qwen_lm, [ITEM.prompt])
    for name, build in CONDITIONS.items():
        interventions = build(js, ITEM, enc, item_rng(0, ITEM, name))
        assert len(interventions) == (0 if name == "none" else 1), name
        assert all(set(iv.layers) <= set(js.band) for iv in interventions)


def test_bridge_clamp_builds_one_readout_clamp_on_the_band(qwen_lm):
    js = synthetic_js(qwen_lm)
    js.readout_stats = synthetic_stats(js)
    enc = encode(qwen_lm, [ITEM.prompt])
    interventions = CONDITIONS["bridge_clamp"](js, ITEM, enc, item_rng(0, ITEM, "bridge_clamp"))
    assert len(interventions) == 1 and isinstance(interventions[0], ReadoutClamp)
    assert interventions[0].layers == js.band


def test_clamp_conditions_without_stats_say_how_to_get_them(qwen_lm):
    js = synthetic_js(qwen_lm)
    enc = encode(qwen_lm, [ITEM.prompt])
    for name in ("bridge_clamp", "wrong_bridge_clamp"):
        with pytest.raises(ValueError, match="run_readout_stats"):
            CONDITIONS[name](js, ITEM, enc, item_rng(0, ITEM, name))


def test_wrong_bridge_clamp_caps_the_swap_to_tokens(qwen_lm):
    js = synthetic_js(qwen_lm)
    js.readout_stats = synthetic_stats(js)
    enc = encode(qwen_lm, [ITEM.prompt])
    right, wrong = (CONDITIONS[c](js, ITEM, enc, item_rng(0, ITEM, c))[0]
                    for c in ("bridge_clamp", "wrong_bridge_clamp"))
    expected = js.readout_stats.caps(token_variants(qwen_lm.tokenizer, ITEM.swap_to), js.band, js.cfg.clamp_z)
    layer = js.band[0]
    torch.testing.assert_close(wrong.caps[layer].cpu(), expected[layer])
    assert right.caps[layer].shape != wrong.caps[layer].shape or not torch.equal(right.caps[layer], wrong.caps[layer])


def test_reference_texts_are_prompts_excluding_their_own_bridge(tokenizer):
    other = TwoHopItem(name="y-0", category="test", prompt="Fact: The capital of Italy is",
                       bridge="Italy", answers=("Rome",))
    texts, exclude = reference_texts(tokenizer, [ITEM, other])
    assert texts == [ITEM.prompt, other.prompt]
    assert exclude == [token_variants(tokenizer, "France"), token_variants(tokenizer, "Italy")]


def test_random_builders_repeat_with_the_same_item_rng(qwen_lm):
    js = synthetic_js(qwen_lm)
    enc = encode(qwen_lm, [ITEM.prompt])
    first, again = (CONDITIONS["random_dir"](js, ITEM, enc, item_rng(0, ITEM, "random_dir"))[0] for _ in range(2))
    torch.testing.assert_close(first.bases[1], again.bases[1])


def test_interventions_for_item_rebuilds_with_a_fresh_rng_every_call(qwen_lm):
    js = synthetic_js(qwen_lm)
    enc = encode(qwen_lm, [ITEM.prompt])
    provider = interventions_for_item(js, ITEM, "random_dir", seed=0)
    first, again = provider(enc)[0], provider(enc)[0]
    fresh = CONDITIONS["random_dir"](js, ITEM, enc, item_rng(0, ITEM, "random_dir"))[0]
    torch.testing.assert_close(first.bases[1], again.bases[1])
    torch.testing.assert_close(first.bases[1], fresh.bases[1])


def test_perplexity_is_token_weighted_over_texts(qwen_lm):
    texts = ["The cat sat on the mat and looked around", "Hi there friend", "x"]
    nll, n = 0.0, 0
    for text in texts[:2]:
        ids = torch.tensor([qwen_lm.token_ids(text)[:6]])
        logits = forward_logits(qwen_lm, ids, torch.ones_like(ids))[0, :-1]
        nll += float(torch.nn.functional.cross_entropy(logits, ids[0, 1:], reduction="sum"))
        n += ids.shape[1] - 1
    with torch.no_grad():
        assert math.isclose(perplexity(qwen_lm, texts, max_len=6, batch_size=2), math.exp(nll / n), rel_tol=1e-4)


def test_is_answer_token():
    assert is_answer_token(" Paris")
    for text in (" ", "\n", "", ' "', " a", " The", ' "the', " called", " number"):
        assert not is_answer_token(text), text


def test_answer_index_skips_filler_quotes_whitespace_and_eos(tokenizer):
    def ids(text):
        return tokenizer(text, add_special_tokens=False).input_ids

    assert tokenizer.decode(ids(' "Paris')[:1]) == ' "'
    assert [tokenizer.decode([t]) for t in ids(" 8")] == [" ", "8"]
    eos = tokenizer.eos_token_id
    assert answer_index(tokenizer, ids(" a calf")) == 1
    assert answer_index(tokenizer, ids(' "Paris')) == 1
    assert answer_index(tokenizer, ids(" 8")) == 1
    assert answer_index(tokenizer, ids(" the number 8")) == 3
    assert answer_index(tokenizer, ids(" Paris")) == 0
    assert answer_index(tokenizer, [eos]) == 0
    assert answer_index(tokenizer, ids(" a") + ids(" the")) == 0
    assert answer_index(tokenizer, ids(" Paris") + [eos]) == 0
