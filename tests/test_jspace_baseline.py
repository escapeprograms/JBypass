"""Readout statistics (jspace.baseline) on tiny models with an identity lens, so each
token's lens logit at layer 1 is W_U[t] . h."""

import dataclasses

import pytest
import torch

from jspace.baseline import collect_readout_stats, load_readout_stats, save_readout_stats
from model.inference import encode, forward_residuals

from conftest import synthetic_js, synthetic_stats

TEXTS = ["The cat sat on the mat today", "Hi there", "Fact: The capital of France is"]


def _direct_stats(js, texts, exclude_ids):
    """Mean, std and count over editable positions, one unpadded text at a time."""
    W_U = js.lm.unembedding_matrix.float()
    skip = js.cfg.ablation_skip_first
    total = torch.zeros(W_U.shape[0], dtype=torch.float64)
    total_sq, count = total.clone(), torch.zeros(W_U.shape[0], dtype=torch.long)
    for text, excluded in zip(texts, exclude_ids):
        resid = forward_residuals(js.lm, encode(js.lm, [text]), [1])[1][0, skip:]
        with torch.no_grad():
            acts = (resid.float() @ W_U.T).double()  # [T, V]
        keep = torch.ones(W_U.shape[0], dtype=torch.bool)
        keep[excluded] = False
        total += acts.sum(0) * keep
        total_sq += (acts ** 2).sum(0) * keep
        count += acts.shape[0] * keep
    mean = total / count
    return mean, (total_sq / count - mean ** 2).clamp(min=0).sqrt(), count


def test_stats_equal_a_direct_computation_over_editable_positions(lm):
    js = synthetic_js(lm)
    exclude = [[5, 17], [], [17]]
    stats = collect_readout_stats(js, TEXTS, exclude, batch_size=2)
    mean, std, count = _direct_stats(js, TEXTS, exclude)
    assert stats.layers == [1] and stats.model == lm.profile.name and stats.n_texts == 3
    assert stats.mean.dtype == torch.float32 and stats.mean.shape == (1, lm.unembedding_matrix.shape[0])
    assert torch.equal(stats.count, count)
    torch.testing.assert_close(stats.mean[0], mean.float(), atol=1e-4, rtol=1e-4)
    torch.testing.assert_close(stats.std[0], std.float(), atol=1e-3, rtol=1e-3)


def test_excluded_ids_contribute_nothing_to_that_tokens_count(qwen_lm):
    js = synthetic_js(qwen_lm)
    token = 42
    positions = [len(qwen_lm.token_ids(t)) - js.cfg.ablation_skip_first for t in TEXTS]
    stats = collect_readout_stats(js, TEXTS, [[token], [], []], batch_size=3)
    assert int(stats.count[token]) == positions[1] + positions[2]
    assert int(stats.count[token + 1]) == sum(positions)
    none_left = collect_readout_stats(js, TEXTS, [[token]] * 3, batch_size=3)
    assert int(none_left.count[token]) == 0
    with pytest.raises(ValueError, match="no reference positions"):
        none_left.caps([token], [1], z=2.0)


def test_caps_are_mean_plus_z_std_and_reject_unknown_layers(qwen_lm):
    stats = synthetic_stats(synthetic_js(qwen_lm))
    caps = stats.caps([3, 9], [1], z=2.0)
    torch.testing.assert_close(caps[1], stats.mean[0, [3, 9]] + 2.0 * stats.std[0, [3, 9]])
    with pytest.raises(ValueError, match="layers"):
        stats.caps([3], [0, 1], z=2.0)


def test_save_load_roundtrip(qwen_lm, tmp_path):
    js = synthetic_js(qwen_lm)
    stats = synthetic_stats(js)
    path = tmp_path / "nested" / "stats.pt"
    save_readout_stats(stats, path)
    loaded = load_readout_stats(path, qwen_lm, js.band)
    assert loaded.layers == stats.layers and loaded.model == stats.model and loaded.n_texts == stats.n_texts
    for field in ("mean", "std", "count"):
        assert torch.equal(getattr(loaded, field), getattr(stats, field))


def test_load_rejects_another_model(qwen_lm, tmp_path):
    js = synthetic_js(qwen_lm)
    path = tmp_path / "stats.pt"
    save_readout_stats(dataclasses.replace(synthetic_stats(js), model="other-model"), path)
    with pytest.raises(ValueError, match="model"):
        load_readout_stats(path, qwen_lm, js.band)


def test_load_rejects_another_vocab_size(qwen_lm, tmp_path):
    js = synthetic_js(qwen_lm)
    stats = synthetic_stats(js)
    path = tmp_path / "stats.pt"
    save_readout_stats(dataclasses.replace(stats, mean=stats.mean[:, :10], std=stats.std[:, :10]), path)
    with pytest.raises(ValueError, match="vocab"):
        load_readout_stats(path, qwen_lm, js.band)


def test_load_rejects_a_missing_band_layer(qwen_lm, tmp_path):
    js = synthetic_js(qwen_lm)
    path = tmp_path / "stats.pt"
    save_readout_stats(synthetic_stats(js), path)
    with pytest.raises(ValueError, match=r"band layers \[2\]"):
        load_readout_stats(path, qwen_lm, [1, 2])
