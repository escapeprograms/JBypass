"""Model-free tests for jspace/ and evaluation/ (synthetic data with known answers).

python -m pytest tests/test_jspace_eval.py
"""

import types

import torch

from evaluation.config import EvalConfig
from evaluation.data_loading import load_dataset
from evaluation.probes import chance_level, probe_accuracy_by_layer
from evaluation.scoring import matches
from evaluation.splits import split_by_bridge
from jspace.config import JSpaceConfig
from jspace.decompose import jspace_decompose
from jspace.interventions import LensSwap, _project_out


def test_datasets_load_and_strip_trailing_space():
    cfg = EvalConfig()
    for name, path in cfg.datasets.items():
        items = load_dataset(name, path)
        assert items and all(not it.prompt.endswith(" ") for it in items)


def test_split_by_bridge_has_no_leakage():
    items = load_dataset("jbypass", EvalConfig().datasets["jbypass"])
    splits = split_by_bridge(items, 0.3, 0)
    bridges = {k: {it.bridge for it in items if it.name in set(v)} for k, v in splits.items()}
    assert not bridges["train"] & bridges["test_unseen"]
    assert bridges["test_seen"] <= bridges["train"]


def test_matches_handles_synonyms_and_filler():
    assert matches(" 8 legs.", ["8", "eight"])
    assert matches(" eight.", ["8", "eight"])
    assert matches(" a calf.", ["calf"])
    assert not matches(" 80", ["8"])


def test_probe_separable_vs_random():
    torch.manual_seed(0)
    labels = [f"b{i % 10}" for i in range(80)]
    groups = [f"s{i}" for i in range(80)]
    centers = torch.randn(10, 64) * 3
    X = torch.stack([centers[int(b[1:])] + torch.randn(64) for b in labels])[:, None].repeat(1, 2, 1)
    X[:, 1] = torch.randn(80, 64)
    acc = probe_accuracy_by_layer(X, labels, groups, 2, 300, 1e-3)
    assert acc[0] > 0.95 and acc[1] < 0.3


def test_probe_folds_block_subject_leakage():
    torch.manual_seed(0)
    subj = torch.randn(12, 32) * 3
    labels = [f"b{s % 3}" for s in range(12) for _ in range(4)]
    groups = [f"s{s}" for s in range(12) for _ in range(4)]
    X = torch.stack([subj[s] + 0.1 * torch.randn(32) for s in range(12) for _ in range(4)])[:, None]
    acc = probe_accuracy_by_layer(X, labels, groups, 2, 300, 1e-3)[0]
    assert acc < chance_level(labels, groups, 2) + 0.25


def test_decompose_recovers_sparse_nonnegative_combination():
    torch.manual_seed(0)
    W_U = torch.randn(500, 64)
    lens = types.SimpleNamespace(jacobians={0: torch.eye(64)})
    ids, coeffs = torch.tensor([3, 77, 210]), torch.tensor([2.0, 1.5, 3.0])
    h = (coeffs[:, None] * W_U[ids]).sum(0, keepdim=True)
    part, support, c = jspace_decompose(lens, W_U, h, 0, k=5)
    assert (h - part).norm() / h.norm() < 0.05
    assert set(ids.tolist()) <= set(support[0].tolist())
    assert (c >= 0).all()


def test_project_out_removes_span():
    v = torch.randn(3, 64)
    Q = torch.linalg.qr(v.T).Q
    assert (_project_out(torch.randn(5, 7, 64), Q) @ v.T).abs().max() < 1e-4


def test_lens_swap_exchanges_coordinates():
    vs, vt = torch.randn(64), torch.randn(64)
    swap = LensSwap(types.SimpleNamespace(cfg=JSpaceConfig()), {0: vs}, {0: vt})
    out = swap.edit((2.0 * vs + 0.5 * vt)[None], 0)[0]
    coords = torch.linalg.lstsq(torch.stack([vs, vt], 1), out[:, None]).solution.squeeze()
    assert torch.allclose(coords, torch.tensor([0.5, 2.0]), atol=1e-3)
