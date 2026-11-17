"""ReadoutClamp and set_readout (jspace.clamp), and load_jspace's readout statistics."""

import dataclasses
import types

import jlens
import pytest
import torch

from jspace.baseline import save_readout_stats
from jspace.clamp import ReadoutClamp, set_readout
from jspace.config import JSpaceConfig
from jspace.interventions import _project_out, editable_positions
from jspace.loading import default_stats_path, load_jspace
from model.hooks import apply_interventions, capture_residuals
from model.inference import encode, forward_logits

from conftest import synthetic_stats

PROMPTS = ["The cat sat on the mat today", "Hi"]
D = 16


def _vectors(k=3, seed=0):
    return torch.randn(k, D, generator=torch.Generator().manual_seed(seed))


def _clamp_setup():
    """A ReadoutClamp on layer 0, residuals h [40, D], and the caps (median readouts),
    with one extra row whose readouts all sit 1 below their caps."""
    V = _vectors()
    h = torch.randn(40, D, generator=torch.Generator().manual_seed(1))
    cap = (h @ V.T).median(dim=0).values
    low = set_readout(h[:1], V, torch.linalg.pinv(V), cap[None] - 1)
    clamp = ReadoutClamp(types.SimpleNamespace(cfg=JSpaceConfig()), {0: V}, {0: cap})
    return clamp, V, torch.cat([h, low]), cap


def test_set_readout_with_zero_targets_is_project_out():
    V = _vectors()
    h = torch.randn(2, 5, D, generator=torch.Generator().manual_seed(1))
    out = set_readout(h, V, torch.linalg.pinv(V), torch.zeros(2, 5, 3))
    torch.testing.assert_close(out, _project_out(h, torch.linalg.qr(V.T).Q), atol=1e-5, rtol=1e-5)


def test_clamp_sets_readouts_to_min_of_readout_and_cap():
    clamp, V, h, cap = _clamp_setup()
    before = h @ V.T
    after = clamp.edit(h, 0) @ V.T
    torch.testing.assert_close(after, torch.minimum(before, cap), atol=1e-4, rtol=1e-4)


def test_clamp_leaves_rows_below_their_caps_bit_identical():
    clamp, V, h, cap = _clamp_setup()
    below = ((h @ V.T) <= cap).all(-1)
    out = clamp.edit(h, 0)
    assert below[-1] and not below.all()
    assert torch.equal(out[below], h[below])
    assert not torch.equal(out[~below], h[~below])


def test_clamp_leaves_the_orthogonal_complement_unchanged():
    clamp, V, h, _ = _clamp_setup()
    complement = torch.eye(D) - torch.linalg.pinv(V) @ V
    torch.testing.assert_close(clamp.edit(h, 0) @ complement, h @ complement, atol=1e-5, rtol=1e-5)


def _layer1(lm, enc, interventions):
    with torch.no_grad(), apply_interventions(lm, interventions, enc.attention_mask), \
            capture_residuals(lm, [1]) as resids:
        forward_logits(lm, enc.input_ids, enc.attention_mask)
    return resids[1]


def test_clamp_on_left_padded_batch_skips_first_token_and_padding(lm):
    vectors = torch.randn(3, lm.d_model, generator=torch.Generator().manual_seed(0))
    js = types.SimpleNamespace(lm=lm, cfg=JSpaceConfig())
    enc = encode(lm, PROMPTS)
    clean = _layer1(lm, enc, [])
    edited = _layer1(lm, enc, [ReadoutClamp(js, {1: vectors}, {1: torch.zeros(3)})])
    editable = editable_positions(enc.attention_mask, js.cfg.ablation_skip_first)
    assert (enc.attention_mask == 0).any()
    assert (edited[editable] @ vectors.T).max() < 1e-4
    assert torch.equal(edited[~editable], clean[~editable])
    first_real = enc.attention_mask.argmax(-1)
    assert (clean[torch.arange(2), first_real] @ vectors.T).max() > 1e-3


def _lens_file(lm, tmp_path):
    path = tmp_path / "lens.pt"
    d = lm.d_model
    jlens.JacobianLens({0: torch.eye(d), 1: torch.eye(d)}, n_prompts=1, d_model=d).save(str(path))
    return str(path)


def test_load_jspace_with_a_missing_explicit_stats_path_raises(qwen_lm, tmp_path):
    with pytest.raises(FileNotFoundError):
        load_jspace(qwen_lm, lens_path=_lens_file(qwen_lm, tmp_path), stats_path=tmp_path / "none.pt")


def test_load_jspace_without_stats_file_has_no_readout_stats(qwen_lm, tmp_path):
    cfg = dataclasses.replace(JSpaceConfig(), readout_stats_dir=tmp_path / "stats")
    js = load_jspace(qwen_lm, cfg, lens_path=_lens_file(qwen_lm, tmp_path))
    assert js.readout_stats is None


def test_load_jspace_reads_the_default_stats_file(qwen_lm, tmp_path):
    cfg = dataclasses.replace(JSpaceConfig(), readout_stats_dir=tmp_path / "stats")
    lens_path = _lens_file(qwen_lm, tmp_path)
    js = load_jspace(qwen_lm, cfg, lens_path=lens_path)
    stats = synthetic_stats(js)
    save_readout_stats(stats, default_stats_path(qwen_lm, cfg))
    assert default_stats_path(qwen_lm, cfg) == tmp_path / "stats" / "tiny-qwen3.pt"
    loaded = load_jspace(qwen_lm, cfg, lens_path=lens_path).readout_stats
    assert loaded.layers == js.band and torch.equal(loaded.mean, stats.mean)
