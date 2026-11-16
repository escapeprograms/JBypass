from dataclasses import replace

import torch

from model.config import GenerationSettings
from model.inference import (
    encode, forward_logits, forward_residuals, generate, greedy_tokens, next_token_logits, sequence_loss,
    sequence_nll,
)
from model.loading import LanguageModel
from model.prompts import chat_prompt

from conftest import GENERATION, QWEN_PROFILE, VOCAB, tiny_qwen3
from dummies import Replace, ZeroOut

PROMPTS = ["The cat sat on the", "Hi", "A much longer prompt that needs no padding at all here"]


def test_chat_prompt_uses_profile_template_kwargs(qwen_lm):
    text = chat_prompt(qwen_lm, "How many legs?")
    assert text.startswith("<|im_start|>user\nHow many legs?<|im_end|>")
    assert text.endswith("<|im_start|>assistant\n<think>\n\n</think>\n\n")


def test_chat_prompt_reads_template_kwargs_from_profile(tokenizer):
    no_switch = replace(QWEN_PROFILE, chat_template_kwargs={})
    lm = LanguageModel(tiny_qwen3(), tokenizer, no_switch)
    assert "<think>" not in chat_prompt(lm, "Q")   # Qwen3's template does read the switch


def test_extra_template_kwargs_are_harmless_when_template_ignores_them(tokenizer):
    template = "{% for m in messages %}[{{ m['content'] }}]{% endfor %}"
    messages = [{"role": "user", "content": "Q"}]
    plain = tokenizer.apply_chat_template(messages, chat_template=template, tokenize=False)
    extra = tokenizer.apply_chat_template(messages, chat_template=template, tokenize=False, enable_thinking=False)
    assert plain == extra == "[Q]"


def test_encode_left_pads(lm):
    enc = encode(lm, PROMPTS)
    assert enc.attention_mask[:, -1].all()
    assert (enc.attention_mask[1, 0] == 0) and (enc.input_ids[1, 0] == lm.tokenizer.pad_token_id)


def test_batched_and_single_next_token_logits_agree(lm):
    batched = next_token_logits(lm, PROMPTS)
    single = torch.cat([next_token_logits(lm, [p]) for p in PROMPTS])
    assert batched.shape == (len(PROMPTS), VOCAB)
    torch.testing.assert_close(batched, single, atol=1e-4, rtol=1e-4)


def test_batched_and_single_sequence_loss_agree(lm):
    texts = ["The cat sat on the", "Hi there", "A much longer prompt that needs no padding at all here"]
    batched = sequence_loss(lm, texts)
    single = torch.cat([sequence_loss(lm, [t]) for t in texts])
    assert batched.shape == (len(texts),)
    torch.testing.assert_close(batched, single, atol=1e-4, rtol=1e-4)


def test_batched_and_single_generate_agree(lm):
    batched = generate(lm, PROMPTS, GENERATION)
    single = [generate(lm, [p], GENERATION)[0] for p in PROMPTS]
    assert batched == single


def test_interventions_change_inference_outputs(lm):
    zero = [ZeroOut(layers=[lm.n_layers - 1])]
    assert not torch.allclose(next_token_logits(lm, PROMPTS[:1], zero), next_token_logits(lm, PROMPTS[:1]))


def test_forward_residuals_shapes_without_interventions(lm):
    enc = encode(lm, PROMPTS)
    resids = forward_residuals(lm, enc, [0, lm.n_layers - 1])
    assert sorted(resids) == [0, lm.n_layers - 1]
    assert all(r.shape == (*enc.input_ids.shape, lm.d_model) for r in resids.values())
    assert not any(r.requires_grad for r in resids.values())


def test_forward_residuals_capture_post_intervention(lm):
    enc = encode(lm, PROMPTS[:1])
    resids = forward_residuals(lm, enc, [0, 1], [ZeroOut(layers=[1])])
    assert torch.equal(resids[1], torch.zeros_like(resids[1]))
    assert resids[0].abs().max() > 0


def _force_eos_after(lm, prompt_len: int, n_tokens: int) -> Replace:
    """Make EOS the argmax once the sequence has prompt_len + n_tokens tokens."""
    u = torch.ones(lm.d_model)
    u[1::2] = -1   # zero mean, unit RMS: both RMSNorm and LayerNorm (at init) leave it as is
    with torch.no_grad():
        lm.unembedding_matrix[lm.tokenizer.eos_token_id] = 10 * u
    return Replace(layers=[lm.n_layers - 1], fn=lambda r: u.expand_as(r).clone(), from_len=prompt_len + n_tokens)


def test_generate_returns_only_new_text_and_stops_at_eos(lm):
    prompt = PROMPTS[0]
    prompt_len = len(lm.token_ids(prompt))
    stopper = _force_eos_after(lm, prompt_len, 2)
    first_two = generate(lm, [prompt], GenerationSettings(max_new_tokens=2))[0]
    stopped = generate(lm, [prompt], GenerationSettings(max_new_tokens=10), [stopper])[0]
    assert stopped == first_two
    assert len(lm.token_ids(stopped)) <= 2
    assert prompt not in stopped


def test_greedy_tokens_calls_provider_once_per_step_with_grown_sequence(lm):
    enc = encode(lm, PROMPTS[:2])
    seen = []
    def provider(step_enc):
        seen.append(step_enc)
        return ()
    tokens = greedy_tokens(lm, enc, 3, provider)
    assert len(seen) == tokens.shape[1] == 3
    widths = [s.input_ids.shape[1] for s in seen]
    assert widths == [enc.input_ids.shape[1] + i for i in range(3)]
    torch.testing.assert_close(seen[-1].input_ids[:, -2:], tokens[:, :2])
    assert all(s.attention_mask.shape == s.input_ids.shape for s in seen)


def _reference_greedy(lm, prompts, n_tokens, interventions):
    """The pre-refactor generation loop: a fixed intervention list, re-run every step."""
    enc = encode(lm, prompts)
    ids, mask = enc.input_ids, enc.attention_mask
    with torch.no_grad():
        for _ in range(n_tokens):
            next_ids = forward_logits(lm, ids, mask, interventions)[:, -1].argmax(-1)
            ids = torch.cat([ids, next_ids[:, None]], dim=1)
            mask = torch.cat([mask, torch.ones_like(next_ids[:, None])], dim=1)
    rows = ids[:, enc.input_ids.shape[1]:].tolist()
    return [lm.tokenizer.decode(row, skip_special_tokens=True) for row in rows]


def test_generate_output_is_unchanged(lm):
    zero = [ZeroOut(layers=[1])]
    assert generate(lm, PROMPTS, GENERATION) == _reference_greedy(lm, PROMPTS, GENERATION.max_new_tokens, ())
    assert generate(lm, PROMPTS, GENERATION, zero) == _reference_greedy(lm, PROMPTS, GENERATION.max_new_tokens, zero)


def test_sequence_nll_sums_match_sequence_loss_and_truncation(lm):
    texts = ["The cat sat on the", "Hi there"]
    total, count = sequence_nll(lm, encode(lm, texts))
    torch.testing.assert_close(total / count, sequence_loss(lm, texts))
    assert count.tolist() == [len(lm.token_ids(t)) - 1 for t in texts]
    assert encode(lm, texts, max_length=3).attention_mask.sum(-1).tolist() == [3, 2]