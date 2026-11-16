"""viz core (no gradio, no network): config, catalog, session, run, slice and page."""

import dataclasses
import gc
import html
import json
import random
import re
import weakref
from pathlib import Path

import jlens.vis
import numpy as np
import pytest
import torch

from conftest import VOCAB, synthetic_js
from evaluation.conditions import CONDITIONS, interventions_for_item
from evaluation.config import EvalConfig
from evaluation.data_loading import TwoHopItem
from evaluation.scoring import matches
from model.cli import DEFAULT_RUN_CONFIG
from model.config import load_run_config
from model.inference import encode, greedy_tokens
from model.lora import PROFILE_FILE
from viz.catalog import (ALL_SPLITS, ModelChoice, examples, lens_profiles, model_choices,
                         pick_random, split_names, valid_conditions, valid_fields)
from viz.config import DEFAULT_VIZ_CONFIG, load_viz_config
from viz.page import iframe, item_card, one_line, slice_page, summary
from viz.run import run_example
from viz.session import Session
from viz.slice import pinned_tokens, replay_mismatches, run_slice, slice_from_ids

PROFILES_DIR = load_run_config(DEFAULT_RUN_CONFIG).profiles_dir
PROMPT = "The capital of France is"
N_NEW = 3


def make_item(**overrides) -> TwoHopItem:
    fields = dict(name="t1", category="geo", prompt=PROMPT, bridge="France", answers=("Paris",),
                  swap_to="Germany", hop1_prompt="France's neighbour is", hop2_prompt=None)
    return TwoHopItem(**{**fields, **overrides})


def write(path: Path, text: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


# ---- config ---------------------------------------------------------------------------------

def test_viz_config_resolves_paths_against_its_folder(tmp_path):
    cfg = load_viz_config(write(tmp_path / "c" / "viz.yaml", (
        "runs_dir: ../runs\nsplit_file: s.json\ntop_n: 5\nmax_tracked: null\n"
        "page_height: 100\nserver_name: localhost\nserver_port: 1\n")))
    assert cfg.runs_dir == (tmp_path / "runs").resolve()
    assert cfg.split_file == (tmp_path / "c" / "s.json").resolve()
    assert cfg.max_tracked is None and cfg.top_n == 5


def test_viz_config_null_split_file(tmp_path):
    cfg = load_viz_config(write(tmp_path / "viz.yaml", (
        "runs_dir: r\nsplit_file: null\ntop_n: 5\nmax_tracked: 3\n"
        "page_height: 100\nserver_name: localhost\nserver_port: 1\n")))
    assert cfg.split_file is None


def test_committed_viz_config_loads():
    cfg = load_viz_config(DEFAULT_VIZ_CONFIG)
    assert cfg.runs_dir.name == "runs" and cfg.top_n > 0 and cfg.server_port == 7860


# ---- catalog --------------------------------------------------------------------------------

def test_lens_profiles_lists_only_profiles_with_a_prefit_lens():
    assert lens_profiles(PROFILES_DIR) == ["qwen3-1.7b"]


def _adapter_dir(runs: Path, name: str, profile: str | None) -> Path:
    d = runs / name
    d.mkdir(parents=True)
    if profile is not None:
        write(d / PROFILE_FILE, json.dumps({"profile": profile}))
    return d


def test_model_choices_keeps_only_adapters_of_lens_profiles(tmp_path):
    runs = tmp_path / "runs"
    kept = _adapter_dir(runs, "mexp", "qwen3-1.7b")
    _adapter_dir(runs, "small", "qwen3-0.6b")
    _adapter_dir(runs, "empty", None)
    choices = model_choices(PROFILES_DIR, runs)
    assert choices == [ModelChoice("qwen3-1.7b"), ModelChoice("qwen3-1.7b", kept)]
    assert [c.label for c in choices] == ["qwen3-1.7b", "qwen3-1.7b + mexp"]


def test_model_choices_without_runs_dir(tmp_path):
    assert model_choices(PROFILES_DIR, tmp_path / "missing") == [ModelChoice("qwen3-1.7b")]


def test_split_names_offers_only_non_empty_splits(tmp_path):
    split_file = write(tmp_path / "s.json", json.dumps({"a": ["t1", "x"], "b": ["x"], "c": []}))
    items = [make_item(name="t1"), make_item(name="t2")]
    assert split_names(items, split_file) == [ALL_SPLITS, "a"]
    assert split_names(items, None) == [ALL_SPLITS]


def test_valid_conditions_and_fields():
    with_swap, without = make_item(), make_item(swap_to=None)
    assert "wrong_bridge" in valid_conditions(with_swap, CONDITIONS)
    assert valid_conditions(without, CONDITIONS) == [c for c in CONDITIONS if c != "wrong_bridge"]
    fields = ("prompt", "hop2_prompt", "hop1_prompt")
    assert valid_fields(with_swap, fields) == ["prompt", "hop1_prompt"]


def test_pick_random_is_seeded():
    items = [make_item(name=f"t{i}") for i in range(20)]
    assert pick_random(items, random.Random(3)) == pick_random(items, random.Random(3))


def test_examples_drops_unreadable_bridges_and_filters_splits(tmp_path, qwen_lm):
    rows = [{"name": "ok", "prompt": PROMPT, "target": "Paris", "intermediates": ["France"]},
            {"name": "bad", "prompt": PROMPT, "target": "Paris", "intermediates": ["Xqzvbnmplk"]},
            {"name": "other", "prompt": PROMPT, "target": "Berlin", "intermediates": ["Germany"]}]
    data = write(tmp_path / "d.json", json.dumps({"items": rows}))
    split_file = write(tmp_path / "s.json", json.dumps({"one": ["ok", "bad"]}))
    cfg = EvalConfig(datasets={"paper_multihop": data})
    js = synthetic_js(qwen_lm)
    assert [it.name for it in examples(js, "paper_multihop", ALL_SPLITS, cfg, split_file)] == ["ok", "other"]
    assert [it.name for it in examples(js, "paper_multihop", "one", cfg, split_file)] == ["ok"]


# ---- session --------------------------------------------------------------------------------

class _Loaded:
    def __init__(self, choice):
        self.choice = choice


def test_session_caches_one_model_and_frees_the_old_one():
    calls = []

    def load(choice):
        calls.append(choice)
        return _Loaded(choice)

    session = Session(load)
    a, b = ModelChoice("m"), ModelChoice("m", Path("adapter"))
    first = session.get(a)
    assert session.get(a) is first and calls == [a]
    old = weakref.ref(first)
    del first
    assert session.get(b).choice == b and calls == [a, b]
    gc.collect()
    assert old() is None


# ---- run ------------------------------------------------------------------------------------

def test_run_example_matches_greedy_tokens_and_scoring(qwen_lm):
    js, item = synthetic_js(qwen_lm), make_item()
    result = run_example(js, item, "prompt", "bridge", 0, N_NEW)
    enc = encode(qwen_lm, [PROMPT])
    direct = greedy_tokens(qwen_lm, enc, N_NEW, interventions_for_item(js, item, "bridge", 0))
    assert result.response_ids == tuple(direct[0].tolist())
    assert result.prompt_ids == tuple(enc.input_ids[0].tolist())
    assert result.correct == matches(result.response_text, result.answers)
    assert all(isinstance(t, int) for t in result.prompt_ids + result.response_ids)


# ---- slice ----------------------------------------------------------------------------------

def test_slice_shapes_and_dtypes(qwen_lm):
    js = synthetic_js(qwen_lm)
    result = run_example(js, make_item(), "prompt", "none", 0, N_NEW)
    data = run_slice(js, result, top_n=5, max_tracked=7)
    seq, n_layers = len(result.prompt_ids) + len(result.response_ids), len(data.layers)
    assert data.layers == [0, 1, 2] and data.seq_len == seq and data.ctx_offset == 0
    assert data.top_ids.shape == data.top_ranks.shape == (seq, n_layers, 5)
    assert data.rank_tensor.shape == (seq, n_layers, len(data.tracked_token_ids))
    assert {a.dtype for a in (data.top_ids, data.top_ranks, data.rank_tensor)} == {np.dtype(np.int32)}
    assert (data.top_ranks == np.arange(5)).all()
    assert data.vocab_size == VOCAB


def test_slice_matches_jlens_compute_slice_on_a_clean_prompt(lm):
    js = synthetic_js(lm)
    enc = encode(lm, [PROMPT])
    assert torch.equal(js.lens_model.encode(PROMPT).cpu(), enc.input_ids.cpu())
    pinned = [12095, 9625]
    ours = slice_from_ids(js, enc, (), 5, 6, pinned)
    ref = jlens.vis.compute_slice(js.lens_model, js.lens, PROMPT, top_n=5, max_tracked=6,
                                  pinned_token_ids=set(pinned))
    assert ours.layers == ref.layers
    assert ours.context_token_strs == ref.context_token_strs
    assert np.array_equal(ours.top_ids, ref.top_ids) and np.array_equal(ours.top_ranks, ref.top_ranks)
    assert ours.tracked_token_ids == ref.tracked_token_ids
    assert np.array_equal(ours.rank_tensor, ref.rank_tensor)
    assert ours.vocab_fragment == ref.vocab_fragment and ours.vocab_size == ref.vocab_size
    assert ours.pinned_token_ids == ref.pinned_token_ids


@pytest.mark.parametrize("condition", list(CONDITIONS))
def test_replay_reproduces_generation_under_every_condition(lm, condition):
    js = synthetic_js(lm)
    result = run_example(js, make_item(), "prompt", condition, 0, N_NEW)
    assert replay_mismatches(run_slice(js, result, 3, 4), result) == []


def test_pins_are_tracked_and_max_tracked_caps_the_rest(qwen_lm):
    js = synthetic_js(qwen_lm)
    result = run_example(js, make_item(), "prompt", "none", 0, N_NEW)
    pins = pinned_tokens(js, result)
    assert pins and pins == sorted(pins)
    data = run_slice(js, result, top_n=5, max_tracked=4)
    assert set(pins) <= set(data.tracked_token_ids)
    assert len(set(data.tracked_token_ids) - set(pins)) <= 4
    assert data.pinned_token_ids == pins


def test_bridge_ablation_changes_the_slice(qwen_lm):
    js = synthetic_js(qwen_lm)
    clean = run_example(js, make_item(), "prompt", "none", 0, N_NEW)
    ablated = run_example(js, make_item(), "prompt", "bridge", 0, N_NEW)
    a, b = run_slice(js, clean, 5, 6), run_slice(js, ablated, 5, 6)
    n = len(clean.prompt_ids)
    assert not np.array_equal(a.top_ids[:n], b.top_ids[:n])


def test_replay_mismatches_reports_a_wrong_token(qwen_lm):
    js = synthetic_js(qwen_lm)
    result = run_example(js, make_item(), "prompt", "none", 0, N_NEW)
    data = run_slice(js, result, 3, 4)
    data.top_ids[len(result.prompt_ids), -1, 0] += 1
    assert replay_mismatches(data, result) == [1]


# ---- page -----------------------------------------------------------------------------------

def test_iframe_srcdoc_round_trips():
    page = '<html><body><script>if (a < b && c > "d") {}</script>\'x\'</body></html>'
    srcdoc = re.search(r'srcdoc="([^"]*)"', iframe(page, 321)).group(1)
    assert html.unescape(srcdoc) == page
    assert "height:321px" in iframe(page, 321)


def _bootstrap(page: str) -> dict:
    start = page.index('{"mode": "embed"')
    return json.JSONDecoder().raw_decode(page, start)[0]


def test_slice_page_embeds_the_slice_without_network(qwen_lm, monkeypatch):
    # The embed template is memoised per mode in _TEMPLATE_FOR_MODE; seeding it skips the d3 fetch.
    monkeypatch.setitem(jlens.vis._TEMPLATE_FOR_MODE, "embed",
                        jlens.vis.PAGE_TEMPLATE.replace("__D3__", "<script>/*d3*/</script>"))
    js = synthetic_js(qwen_lm)
    result = run_example(js, make_item(), "prompt", "bridge", 0, N_NEW)
    data = run_slice(js, result, 3, 4)
    meta = _bootstrap(slice_page(data, result, "tiny-qwen3"))["meta"]
    assert meta["T"] == len(result.prompt_ids) + len(result.response_ids)
    assert meta["pinned"] == pinned_tokens(js, result)
    assert "bridge" in meta["what"] and meta["title"] == "tiny-qwen3"


def test_summary_mentions_response_and_mismatches(qwen_lm):
    result = run_example(synthetic_js(qwen_lm), make_item(), "prompt", "none", 0, N_NEW)
    n = len(result.prompt_ids)
    clean = summary(result, [])
    assert one_line(result.response_text) in clean and "reproduces generation" in clean
    assert one_line("a\nb") == "a⏎b"
    assert f"positions {n}-{n + N_NEW - 1}" in clean
    assert f"positions {n + 1}" in summary(result, [1])


# ---- app (gradio) ---------------------------------------------------------------------------

def _app_context(js, tmp_path):
    from viz.app import AppContext
    rows = [{"name": "ok", "prompt": PROMPT, "target": "Paris", "intermediates": ["France"]}]
    cfg = EvalConfig(datasets={"paper_multihop": write(tmp_path / "d.json", json.dumps({"items": rows}))})
    viz = dataclasses.replace(load_viz_config(DEFAULT_VIZ_CONFIG), split_file=None, top_n=3, max_tracked=4)
    return AppContext(run=load_run_config(DEFAULT_RUN_CONFIG), viz=viz, eval_cfg=cfg,
                      session=Session(lambda choice: js), choices={"tiny": ModelChoice("tiny")},
                      default_model="tiny", rng=random.Random(0))


def _values(dropdown) -> list[str]:
    return [value for _, value in dropdown.choices]


def test_app_builds_and_choice_handlers_filter_invalid_options(qwen_lm, tmp_path):
    gr = pytest.importorskip("gradio")
    from viz.app import build_app, on_dataset, on_example, on_split
    ctx = _app_context(synthetic_js(qwen_lm), tmp_path)
    assert isinstance(build_app(ctx), gr.Blocks)
    assert _values(on_dataset(ctx, "tiny", "paper_multihop")) == [ALL_SPLITS]
    items, example = on_split(ctx, "tiny", "paper_multihop", ALL_SPLITS)
    assert list(items) == ["ok"] and example.value == "ok"
    no_swap = {"t": make_item(swap_to=None)}
    fields, conditions = on_example(ctx, no_swap, "t", "hop2_prompt", "wrong_bridge")
    assert _values(fields) == ["prompt", "hop1_prompt"] and fields.value == "prompt"
    assert "wrong_bridge" not in _values(conditions) and conditions.value == "none"


def test_app_run_handler_returns_summary_and_iframe(qwen_lm, tmp_path, monkeypatch):
    gr = pytest.importorskip("gradio")
    from viz.app import on_run
    monkeypatch.setitem(jlens.vis._TEMPLATE_FOR_MODE, "embed",
                        jlens.vis.PAGE_TEMPLATE.replace("__D3__", "<script>/*d3*/</script>"))
    ctx = _app_context(synthetic_js(qwen_lm), tmp_path)
    items = {"t": make_item()}
    text, embed = on_run(ctx, "tiny", items, "t", "prompt", "bridge")
    assert "**Response** (bridge)" in text and embed.startswith("<iframe") and "tiny: t" in html.unescape(embed)
    with pytest.raises(gr.Error):
        on_run(ctx, "tiny", items, None, "prompt", "none")


def test_item_card_shows_the_fields_prompt_and_answers():
    card = item_card(make_item(), "hop1_prompt")
    assert "France's neighbour is" in card and "**Expected:** France" in card and "Germany" in card
