"""evaluation on the shared model: decoding, condition builders and perplexity
(tiny Qwen-shaped model, synthetic lens, CPU)."""

import math

import torch

from evaluation.conditions import CONDITIONS, item_rng
from evaluation.data_loading import TwoHopItem
from evaluation.perplexity import perplexity
from evaluation.scoring import greedy_continuation
from model.inference import encode, forward_logits, greedy_tokens

from conftest import synthetic_js

ITEM = TwoHopItem(name="x-0", category="test", prompt="Fact: The capital of the country is",
                  bridge="France", answers=("Paris",), swap_to="Spain")


def test_greedy_continuation_returns_text_and_first_token(qwen_lm):
    text, first = greedy_continuation(qwen_lm, ITEM.prompt, lambda enc: (), 3)
    tokens = greedy_tokens(qwen_lm, encode(qwen_lm, [ITEM.prompt]), 3, lambda enc: ())[0].tolist()
    assert text == qwen_lm.tokenizer.decode(tokens) and first == tokens[0]


def test_builders_return_sequences_and_none_is_empty(qwen_lm):
    js = synthetic_js(qwen_lm)
    enc = encode(qwen_lm, [ITEM.prompt])
    for name, build in CONDITIONS.items():
        interventions = build(js, ITEM, enc, item_rng(0, ITEM, name))
        assert len(interventions) == (0 if name == "none" else 1), name
        assert all(set(iv.layers) <= set(js.band) for iv in interventions)


def test_random_builders_repeat_with_the_same_item_rng(qwen_lm):
    js = synthetic_js(qwen_lm)
    enc = encode(qwen_lm, [ITEM.prompt])
    first, again = (CONDITIONS["random_dir"](js, ITEM, enc, item_rng(0, ITEM, "random_dir"))[0] for _ in range(2))
    torch.testing.assert_close(first.bases[1], again.bases[1])


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
