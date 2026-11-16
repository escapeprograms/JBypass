"""jspace on the shared model: interventions through model.hooks, lens_view, jlens's
tokenisation, lens loading and lens readout under interventions (tiny models, CPU)."""

import types

import jlens
import pytest
import torch

from jspace.config import JSpaceConfig
from jspace.interventions import ProjectOut, editable_positions
from jspace.loading import lens_view, load_lens
from jspace.readout import lens_logits, readout_layers, residual_logits
from model.hooks import HookContext, apply_interventions, capture_residuals
from model.inference import encode, forward_logits, forward_residuals
from model.lora import attach_lora

from conftest import LORA, synthetic_js

PROMPTS = ["The cat sat on the mat today", "Hi", "Fact: The capital of France is"]


def _layer1(lm, enc, interventions):
    with torch.no_grad(), apply_interventions(lm, interventions, enc.attention_mask), \
            capture_residuals(lm, [1]) as resids:
        forward_logits(lm, enc.input_ids, enc.attention_mask)
    return resids[1]


def test_project_out_on_left_padded_batch_skips_first_token_and_padding(lm):
    vectors = torch.randn(3, lm.d_model, generator=torch.Generator().manual_seed(0))
    js = types.SimpleNamespace(lm=lm, cfg=JSpaceConfig())
    enc = encode(lm, PROMPTS[:2])
    clean = _layer1(lm, enc, [])
    edited = _layer1(lm, enc, [ProjectOut(js, {1: vectors})])
    editable = editable_positions(enc.attention_mask, js.cfg.ablation_skip_first)
    assert (enc.attention_mask == 0).any()
    assert int(editable.sum()) == int(enc.attention_mask.sum()) - len(PROMPTS[:2])
    assert (edited[editable] @ vectors.T).abs().max() < 1e-4
    assert torch.equal(edited[~editable], clean[~editable])
    first_real = enc.attention_mask.argmax(-1)
    assert (clean[torch.arange(2), first_real] @ vectors.T).abs().max() > 1e-3


def test_unpadded_edit_is_exactly_the_edit_of_the_tail():
    """On one unpadded row the result is bit-identical to editing h[:, skip_first:] alone."""
    js = types.SimpleNamespace(cfg=JSpaceConfig())
    iv = ProjectOut(js, {0: torch.randn(4, 16, generator=torch.Generator().manual_seed(2))})
    h = torch.randn(1, 9, 16, dtype=torch.bfloat16)
    out = iv(h, HookContext(0, torch.ones(1, 9, dtype=torch.long)))
    skip = js.cfg.ablation_skip_first
    expected = torch.cat([h[:, :skip], iv.edit(h[:, skip:].float(), 0).to(h.dtype)], dim=1)
    assert out.dtype == h.dtype and torch.equal(out, expected)


def test_lens_view_restores_grad_flags_and_train_mode(lm):
    def state():
        return ([p.requires_grad for p in lm.base_model.parameters()],
                [m.training for m in lm.base_model.modules()])
    next(lm.base_model.parameters()).requires_grad_(False)
    lm.hf_model.train()
    lm.blocks[0].eval()
    before = state()
    view = lens_view(lm)
    assert state() == before
    assert all(a is b for a, b in zip(view.layers, lm.blocks))
    attach_lora(lm, LORA)
    lm.hf_model.train()
    before = state()
    lens_view(lm)
    assert state() == before and any(before[0])


def test_jlens_encoding_equals_token_ids_for_qwen(qwen_lm):
    view = lens_view(qwen_lm)
    for prompt in PROMPTS:
        assert view.encode(prompt)[0].tolist() == qwen_lm.token_ids(prompt)


def test_lens_logits_apply_interventions_inside_jlens_forward(qwen_lm):
    """With J = I and Qwen's unit-weight RMSNorm at init, the lens logit of token t is
    W_U[t] . h / rms(h), so projecting out W_U[t] makes it zero after the first token."""
    js = synthetic_js(qwen_lm)
    token = qwen_lm.token_ids(" Paris")[0]
    ablate = ProjectOut(js, {1: qwen_lm.unembedding_matrix[[token]].detach()})
    clean = lens_logits(js, PROMPTS[2], layers=[1])[1][:, token]
    ablated = lens_logits(js, PROMPTS[2], [ablate], layers=[1])[1][:, token]
    assert ablated[1:].abs().max() < 1e-4 < clean[1:].abs().max()
    assert ablated[0] == clean[0]


def test_residual_logits_with_identity_lens_is_the_unembedding(lm):
    js = synthetic_js(lm)
    resid = forward_residuals(lm, encode(lm, PROMPTS[:2]), [1])[1]
    logits = residual_logits(js, resid, 1)
    assert logits.dtype == torch.float32
    with torch.no_grad():
        torch.testing.assert_close(logits, lm.unembed(resid))


def test_residual_logits_at_unfitted_final_layer_are_the_model_output(lm):
    final = lm.n_layers - 1
    js = synthetic_js(lm, layers=(0, 1))
    enc = encode(lm, PROMPTS)
    resid = forward_residuals(lm, enc, [final])[final]
    with torch.no_grad():
        expected = forward_logits(lm, enc.input_ids, enc.attention_mask)
    torch.testing.assert_close(residual_logits(js, resid, final), expected)


def test_residual_logits_rejects_an_unfitted_non_final_layer(qwen_lm):
    js = synthetic_js(qwen_lm, layers=(0, 2))
    with pytest.raises(ValueError, match="layer 1"):
        residual_logits(js, torch.zeros(1, 2, qwen_lm.d_model), 1)


def test_readout_layers_are_fitted_layers_plus_final(qwen_lm):
    assert readout_layers(synthetic_js(qwen_lm, layers=(0,))) == [0, qwen_lm.n_layers - 1]
    assert readout_layers(synthetic_js(qwen_lm)) == list(range(qwen_lm.n_layers))


def test_load_lens_rejects_a_lens_for_another_width(qwen_lm, tmp_path):
    path = tmp_path / "lens.pt"
    jlens.JacobianLens({0: torch.eye(32)}, n_prompts=1, d_model=32).save(str(path))
    with pytest.raises(ValueError, match="d_model"):
        load_lens(qwen_lm, str(path))


def test_load_lens_rejects_layers_the_model_lacks(qwen_lm, tmp_path):
    path = tmp_path / "lens.pt"
    d = qwen_lm.d_model
    jlens.JacobianLens({qwen_lm.n_layers: torch.eye(d)}, n_prompts=1, d_model=d).save(str(path))
    with pytest.raises(ValueError, match="layers"):
        load_lens(qwen_lm, str(path))


def test_load_lens_raises_without_lens_path_or_prefit_lens(qwen_lm):
    assert qwen_lm.profile.prefit_lens is None
    with pytest.raises(ValueError, match="prefit_lens"):
        load_lens(qwen_lm)


def test_load_lens_accepts_a_fitting_lens(qwen_lm, tmp_path):
    path = tmp_path / "lens.pt"
    d = qwen_lm.d_model
    jlens.JacobianLens({0: torch.eye(d), 1: torch.eye(d)}, n_prompts=1, d_model=d).save(str(path))
    lens = load_lens(qwen_lm, str(path))
    assert lens.source_layers == [0, 1] and lens.jacobians[0].device == qwen_lm.device
