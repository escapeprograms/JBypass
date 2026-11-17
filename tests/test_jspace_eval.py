"""Model-free tests for jspace/ and evaluation/ (synthetic data with known answers).

python -m pytest tests/test_jspace_eval.py
"""

import json
import re
import types
from pathlib import Path

import pytest
import torch

from evaluation.config import EvalConfig
from evaluation.data.build_jbypass_dataset import (
    BRIDGE_ZH, BRIDGES, OUT, RELATIONS, add_swaps, build_items, check_bridge_zh, fill_relation, validate,
)
from evaluation.data_loading import TwoHopItem, load_dataset
from evaluation.probes import chance_level, probe_accuracy_by_layer
from evaluation.scoring import matches
from evaluation.show_table import describe, expand
from evaluation.splits import split_by_bridge
from jspace.config import JSpaceConfig
from jspace.decompose import jspace_decompose
from jspace.interventions import LensSwap, _project_out
from jspace.vectors import single_token_ids, token_variants


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


def _built_items():
    items = build_items()
    validate(items)
    return add_swaps(items)


def test_dataset_names_unique():
    names = [it["name"] for it in _built_items()]
    assert len(names) == len(set(names))


def test_dataset_bridge_in_one_category():
    categories = {}
    for it in _built_items():
        categories.setdefault(it["bridge"], set()).add(it["category"])
    assert all(len(c) == 1 for c in categories.values())


def test_dataset_bridge_absent_from_two_hop_prompt():
    for it in _built_items():
        assert not re.search(rf"\b{re.escape(it['bridge'])}\b", it["prompt"], re.IGNORECASE), it["name"]


def test_dataset_hop2_is_relation_template_with_bridge_phrase():
    phrase = {b.name: b.phrase for rows in BRIDGES.values() for b in rows}
    for it in _built_items():
        template = RELATIONS[it["category"]][it["relation"]]
        assert it["hop2_prompt"] == fill_relation(template, phrase[it["bridge"]])


def test_dataset_swap_is_other_bridge_of_same_category():
    items = _built_items()
    category_of = {it["bridge"]: it["category"] for it in items}
    for it in items:
        assert it["swap_to"] != it["bridge"]
        assert category_of[it["swap_to"]] == it["category"]


def test_dataset_every_bridge_has_chinese_names():
    check_bridge_zh()
    for it in _built_items():
        assert it["bridge_zh"] and all(s and not s.isascii() for s in it["bridge_zh"]), it["name"]


def test_check_bridge_zh_raises_on_missing_and_unknown():
    missing = {k: v for k, v in BRIDGE_ZH.items() if k != "France"}
    with pytest.raises(ValueError, match="France"):
        check_bridge_zh(table=missing)
    with pytest.raises(ValueError, match="Atlantis"):
        check_bridge_zh(table={**BRIDGE_ZH, "Atlantis": ["亚特兰蒂斯"]})


def test_committed_dataset_matches_the_builder():
    committed = json.loads(Path(OUT).read_text(encoding="utf-8"))
    assert committed == {"items": _built_items()}


def test_load_jbypass_reads_bridge_zh():
    datasets = EvalConfig().datasets
    items = load_dataset("jbypass", datasets["jbypass"])
    assert all(isinstance(it.bridge_zh, tuple) and it.bridge_zh for it in items)
    assert all("法国" in it.bridge_zh for it in items if it.bridge == "France")
    assert all(it.bridge_zh == () for it in load_dataset("paper_probe_swap", datasets["paper_probe_swap"]))
    assert TwoHopItem(name="x", category="c", prompt="p", bridge="b", answers=("a",)).bridge_zh == ()


def test_single_token_ids_keeps_single_tokens_in_order(tokenizer):
    def n_tokens(text):
        return len(tokenizer(text, add_special_tokens=False).input_ids)
    assert n_tokens("爱因斯坦") > 1 and n_tokens(" 法国") > 1
    france = tokenizer("法国", add_special_tokens=False).input_ids[0]
    assert single_token_ids(tokenizer, ["法国", "爱因斯坦", " 法国", "法国"]) == [france]


def test_token_variants_unchanged(tokenizer):
    ids = token_variants(tokenizer, "France")
    assert ids and len(ids) == len(set(ids))
    for i in ids:
        assert tokenizer(tokenizer.decode([i]), add_special_tokens=False).input_ids == [i]


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


def _record(name, field, condition, correct, next_token_correct):
    return {"name": name, "field": field, "condition": condition, "correct": correct,
            "next_token_correct": next_token_correct, "output": ""}


def test_show_table_describe_rebuilds_tables_from_records():
    records = [_record("a", "prompt", "none", True, True), _record("b", "prompt", "none", True, False),
               _record("a", "prompt", "bridge", True, False), _record("b", "prompt", "bridge", False, False),
               _record("a", "hop2_prompt", "none", True, True), _record("b", "hop2_prompt", "none", True, True),
               _record("a", "hop2_prompt", "bridge", False, False), _record("b", "hop2_prompt", "bridge", False, False)]
    result = {"tag": "M", "dataset": "jbypass", "split": None, "adapter": None, "table": {}, "records": records}
    lines = describe("eval_M_jbypass_all.json", result).splitlines()
    assert "base model" in lines[0] and "items=2" in lines[0] and "eval_M_jbypass_all.json" in lines[0]
    exact, paper = lines.index("exact match (primary):"), lines.index("next token (paper's metric):")
    assert lines[exact + 1].split() == ["prompt", "type", "none", "bridge"]
    assert lines[exact + 2].split() == ["prompt", "1.00", "0.50"]
    assert lines[exact + 3].split() == ["hop2_prompt", "1.00", "0.00"]
    assert lines[paper + 2].split() == ["prompt", "0.50", "0.00"]
    assert lines[paper + 3].split() == ["hop2_prompt", "1.00", "0.00"]


def test_show_table_expand_raises_on_unmatched_pattern(tmp_path):
    (tmp_path / "eval_b.json").write_text("{}", encoding="utf-8")
    (tmp_path / "eval_a.json").write_text("{}", encoding="utf-8")
    assert [p.name for p in expand([str(tmp_path / "eval_*.json")])] == ["eval_a.json", "eval_b.json"]
    with pytest.raises(FileNotFoundError):
        expand([str(tmp_path / "missing_*.json")])
