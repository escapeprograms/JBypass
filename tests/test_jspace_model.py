"""jspace on the shared model: interventions through model.hooks, lens_view, jlens's
tokenisation, lens loading, the band load_jspace builds, and lens readout under
interventions (tiny models, CPU)."""

import dataclasses
import logging
import types
from pathlib import Path

import jlens
import pytest
import torch

from jspace.config import JSpaceConfig, jspace_config
from jspace.fit_lens import fit_to_file
from jspace.interventions import ProjectOut, editable_positions
from jspace.loading import default_stats_path, lens_view, load_jspace, load_lens, profile_stats_path
from jspace.readout import lens_logits, readout_layers, residual_logits
from jspace.vectors import layer_range, workspace_band
from model.config import load_run_config
from model.hooks import HookContext, apply_interventions, capture_residuals
from model.inference import encode, forward_logits, forward_residuals
from model.lora import attach_lora

from conftest import LORA, synthetic_js

PROMPTS = ["The cat sat on the mat today", "Hi", "Fact: The capital of France is"]
RUN_YAML = Path(__file__).resolve().parents[1] / "configs" / "run.yaml"


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


FIT_PROMPTS = [
    "The quick brown fox jumps over the lazy dog while the farmer watches from the old wooden porch at dawn.",
    "In the spring of that year the small town council voted to rebuild the bridge that the river had washed away.",
]


def _assert_full_lens_file(lm, path: Path, n_prompts: int) -> jlens.JacobianLens:
    """The file at `path` loads for lm, has n_prompts and a J matrix for every layer below the final one."""
    lens = load_lens(lm, str(path))
    assert lens.source_layers == list(range(lm.n_layers - 1)) and lens.n_prompts == n_prompts
    return lens


@pytest.mark.parametrize("with_lora", [False, True])
def test_fit_to_file_saves_a_lens_for_every_layer_below_the_final(lm, tmp_path, with_lora):
    if with_lora:
        attach_lora(lm, LORA)
    out = tmp_path / "lenses" / "fit.pt"
    lens = fit_to_file(lm, FIT_PROMPTS, JSpaceConfig(), out)
    assert lens.n_prompts == len(FIT_PROMPTS)
    _assert_full_lens_file(lm, out, len(FIT_PROMPTS))
    assert Path(f"{out}.ckpt").exists()


def test_refitting_the_same_out_with_more_prompts_extends_the_fit(qwen_lm, tmp_path, caplog):
    cfg = JSpaceConfig()
    extended, fresh = tmp_path / "extended.pt", tmp_path / "fresh.pt"
    fit_to_file(qwen_lm, FIT_PROMPTS[:1], cfg, extended)
    with caplog.at_level(logging.INFO, logger="jlens.fitting"):
        fit_to_file(qwen_lm, FIT_PROMPTS, cfg, extended)
    # Equal matrices alone would also hold for a refit from scratch; jlens logs the resume.
    assert "resuming from checkpoint: 1/2 prompts processed" in caplog.text
    fit_to_file(qwen_lm, FIT_PROMPTS, cfg, fresh)
    a = _assert_full_lens_file(qwen_lm, extended, 2)
    b = _assert_full_lens_file(qwen_lm, fresh, 2)
    for layer in b.source_layers:
        torch.testing.assert_close(a.jacobians[layer], b.jacobians[layer])


def test_layer_range_is_inclusive():
    assert layer_range(8, 26, 28) == list(range(8, 27))
    assert layer_range(0, 0, 1) == [0]


@pytest.mark.parametrize("first, last", [(-1, 2), (3, 2), (1, 28)])
def test_layer_range_rejects_negative_reversed_and_out_of_model(first, last):
    with pytest.raises(ValueError, match="band layers"):
        layer_range(first, last, 28)


def test_default_fractions_give_layers_11_to_26_on_28_layers():
    cfg = JSpaceConfig()
    assert cfg.band_layers is None
    assert workspace_band(28, cfg.band_start, cfg.band_end) == list(range(11, 27))


def _all_layers_lens(lm, tmp_path, layers=None) -> str:
    """A lens file with identity J matrices at `layers` (default: every layer of lm)."""
    path = tmp_path / "lens.pt"
    layers = range(lm.n_layers) if layers is None else layers
    eye = torch.eye(lm.d_model)
    jlens.JacobianLens({l: eye for l in layers}, n_prompts=1, d_model=lm.d_model).save(str(path))
    return str(path)


def _no_stats(tmp_path, **band) -> JSpaceConfig:
    return dataclasses.replace(JSpaceConfig(), readout_stats_dir=tmp_path / "stats", **band)


def test_profile_stats_path_equals_default_stats_path(qwen_lm, tmp_path):
    cfg = _no_stats(tmp_path)
    expected = tmp_path / "stats" / f"{qwen_lm.profile.name}.pt"
    assert profile_stats_path(qwen_lm.profile.name, cfg) == default_stats_path(qwen_lm, cfg) == expected


def test_band_layers_set_the_band(qwen_lm, tmp_path):
    js = load_jspace(qwen_lm, _no_stats(tmp_path, band_layers=(0, 1)), _all_layers_lens(qwen_lm, tmp_path))
    assert js.band == [0, 1]


def test_band_layers_from_yaml_reach_load_jspace(qwen_lm, tmp_path):
    path = tmp_path / "run.yaml"
    path.write_text(RUN_YAML.read_text(encoding="utf-8")
                    + "jspace:\n  band_layers: [1, 2]\n  readout_stats_dir: stats\n", encoding="utf-8")
    js = load_jspace(qwen_lm, jspace_config(load_run_config(path)), _all_layers_lens(qwen_lm, tmp_path))
    assert js.band == [1, 2]


def test_without_band_layers_the_fractions_set_the_band(qwen_lm, tmp_path):
    cfg = _no_stats(tmp_path, band_start=0.34, band_end=0.6)
    js = load_jspace(qwen_lm, cfg, _all_layers_lens(qwen_lm, tmp_path))
    assert js.band == workspace_band(qwen_lm.n_layers, 0.34, 0.6) == [1, 2]


def test_load_jspace_rejects_a_band_layer_the_lens_lacks(qwen_lm, tmp_path):
    lens_path = _all_layers_lens(qwen_lm, tmp_path, layers=[0, 1])
    with pytest.raises(ValueError, match=r"band layers \[2\]"):
        load_jspace(qwen_lm, _no_stats(tmp_path, band_layers=(1, 2)), lens_path)


def test_load_jspace_rejects_band_layers_past_the_model(qwen_lm, tmp_path):
    with pytest.raises(ValueError, match="3-layer model"):
        load_jspace(qwen_lm, _no_stats(tmp_path, band_layers=(1, 3)), _all_layers_lens(qwen_lm, tmp_path))
