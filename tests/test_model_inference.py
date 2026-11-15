from dataclasses import replace

import torch

from jbypass.model.config import GenerationSettings
from jbypass.model.inference import encode, generate, next_token_logits, sequence_loss
from jbypass.model.loading import LanguageModel
from jbypass.model.prompts import chat_prompt

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
