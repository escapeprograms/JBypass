"""viz core (no gradio, no network): config, catalog, session, run, slice, dictionary, glosses and page."""

import dataclasses
import gc
import gzip
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

from conftest import VOCAB, synthetic_js, synthetic_stats
from evaluation.conditions import CONDITIONS, NEEDS_STATS, interventions_for_item
from evaluation.config import EvalConfig
from evaluation.data_loading import TwoHopItem
from evaluation.scoring import answer_index, answer_token_ids, matches
from jspace import JSpaceConfig, load_jspace, token_variants
from model.cli import DEFAULT_RUN_CONFIG
from model.config import load_run_config
from model.inference import encode, forward_residuals, greedy_tokens
from model.lora import PROFILE_FILE
from viz.cache import ResultCache, RunKey, Shown
from viz.catalog import (ALL_SPLITS, NEEDS_SWAP, ModelChoice, buildable_conditions, examples, fit_command,
                         lens_profiles, model_choices, pick_random, refit_lens_file, split_names,
                         valid_conditions, valid_fields, with_refit_lenses)
from viz.cedict import (CEDICT_FILE, CEDICT_PAGE, Entry, Segment, load_cedict, parse_cedict, parse_line,
                        segment)
from viz.config import DEFAULT_VIZ_CONFIG, load_viz_config
from viz.gloss import gloss_payload, is_foreign, token_gloss
from viz.page import (GLOSS_DATA_ID, iframe, item_card, no_refit_lens_note, one_line, responses_table, slice_page,
                      summary, with_glosses)
from viz.run import run_example
from viz.session import Loaded, Session, attach_refit
from viz.slice import pinned_tokens, replay_mismatches, run_slice, run_slices, shown_encoding, slice_from_ids

PROFILES_DIR = load_run_config(DEFAULT_RUN_CONFIG).profiles_dir
PROMPT = "The capital of France is"
N_NEW = 3

# A dictionary excerpt in both bracket forms (v1 "[pin1 yin1]", v2 "[[pin1yin1]]").
SAMPLE_LINES = [
    "# CC-CEDICT",
    "#! entries=12",
    "",
    "都 都 [dou1] /all/both/",
    "都 都 [Du1] /surname Du/",
    "都 都 [du1] /capital city/metropolis/",
    "已經 已经 [yi3 jing1] /already/",
    "已 已 [yi3] /already/to stop/",
    "經 经 [jing1] /classics/scripture/",
    "好 好 [hao3] /good/well/",
    "好多 好多 [[hao3duo1]] /many; quite a lot/much/",
    "多 多 [[duo1]] /many; much/multi-/",
    "了 了 [le5] /(completed action marker)/",
    "的 的 [[de5]] /of; ~'s (possessive particle)/",
    "中國 中国 [Zhong1 guo2] /China/",
    "後 后 [hou4] /back/behind/",
]
SAMPLE = parse_cedict(SAMPLE_LINES)


def make_item(**overrides) -> TwoHopItem:
    fields = dict(name="t1", category="geo", prompt=PROMPT, bridge="France", answers=("Paris",),
                  swap_to="Germany", hop1_prompt="France's neighbour is", hop2_prompt=None,
                  bridge_zh=("法国",))
    return TwoHopItem(**{**fields, **overrides})


def write(path: Path, text: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


# ---- config ---------------------------------------------------------------------------------

def test_viz_config_resolves_paths_against_its_folder(tmp_path):
    cfg = load_viz_config(write(tmp_path / "c" / "viz.yaml", (
        "runs_dir: ../runs\nlenses_dir: ../lenses\nsplit_file: s.json\ndictionary_file: ../d/cedict.txt.gz\n"
        "top_n: 5\nmax_tracked: null\npage_height: 100\nserver_name: localhost\nserver_port: 1\n")))
    assert cfg.runs_dir == (tmp_path / "runs").resolve()
    assert cfg.lenses_dir == (tmp_path / "lenses").resolve()
    assert cfg.split_file == (tmp_path / "c" / "s.json").resolve()
    assert cfg.dictionary_file == (tmp_path / "d" / "cedict.txt.gz").resolve()
    assert cfg.max_tracked is None and cfg.top_n == 5


def test_viz_config_null_split_and_dictionary_files(tmp_path):
    cfg = load_viz_config(write(tmp_path / "viz.yaml", (
        "runs_dir: r\nlenses_dir: l\nsplit_file: null\ndictionary_file: null\ntop_n: 5\nmax_tracked: 3\n"
        "page_height: 100\nserver_name: localhost\nserver_port: 1\n")))
    assert cfg.split_file is None and cfg.dictionary_file is None


def test_committed_viz_config_loads():
    cfg = load_viz_config(DEFAULT_VIZ_CONFIG)
    assert cfg.runs_dir.name == "runs" and cfg.top_n > 0 and cfg.server_port == 7860
    assert cfg.lenses_dir == cfg.runs_dir.parent / "lenses"
    assert cfg.dictionary_file.name == CEDICT_FILE and cfg.dictionary_file.parent.name == "downloads"


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


def test_refit_lens_file_is_named_after_the_adapter_folder_or_the_profile(tmp_path):
    assert refit_lens_file(ModelChoice("qwen3-1.7b"), tmp_path) == tmp_path / "qwen3-1.7b.pt"
    assert refit_lens_file(ModelChoice("qwen3-1.7b", Path("runs/mclamp")), tmp_path) == tmp_path / "mclamp.pt"


def test_with_refit_lenses_sets_lens_only_where_the_file_exists(tmp_path):
    lens = write(tmp_path / "mclamp.pt", "")
    choices = [ModelChoice("qwen3-1.7b"), ModelChoice("qwen3-1.7b", Path("runs/mclamp")),
               ModelChoice("qwen3-1.7b", Path("runs/mexp"))]
    found = with_refit_lenses(choices, tmp_path)
    assert [c.lens for c in found] == [None, lens, None]
    assert [c.label for c in found] == [c.label for c in choices]


def test_fit_command_has_adapter_only_for_adapters(tmp_path):
    out = tmp_path / "x.pt"
    base = fit_command(ModelChoice("qwen3-1.7b"), out)
    assert base == f"python -m jspace.fit_lens --model qwen3-1.7b --out {out}"
    adapter = fit_command(ModelChoice("qwen3-1.7b", Path("runs/mclamp")), out)
    assert adapter == f"python -m jspace.fit_lens --model qwen3-1.7b --adapter {Path('runs/mclamp')} --out {out}"


def test_split_names_offers_only_non_empty_splits(tmp_path):
    split_file = write(tmp_path / "s.json", json.dumps({"a": ["t1", "x"], "b": ["x"], "c": []}))
    items = [make_item(name="t1"), make_item(name="t2")]
    assert split_names(items, split_file) == [ALL_SPLITS, "a"]
    assert split_names(items, None) == [ALL_SPLITS]


def test_valid_conditions_and_fields():
    with_swap, without = make_item(), make_item(swap_to=None)
    assert "wrong_bridge" in valid_conditions(with_swap, CONDITIONS)
    assert valid_conditions(without, CONDITIONS) == [c for c in CONDITIONS if c not in NEEDS_SWAP]
    fields = ("prompt", "hop2_prompt", "hop1_prompt")
    assert valid_fields(with_swap, fields) == ["prompt", "hop1_prompt"]


def test_buildable_conditions_drop_clamps_only_without_readout_stats():
    item = make_item()
    assert buildable_conditions(False, item, CONDITIONS) == [c for c in CONDITIONS if c not in NEEDS_STATS]
    assert buildable_conditions(True, item, CONDITIONS) == valid_conditions(item, CONDITIONS)


def test_pick_random_is_seeded():
    items = [make_item(name=f"t{i}") for i in range(20)]
    assert pick_random(items, random.Random(3)) == pick_random(items, random.Random(3))


def test_examples_drops_unreadable_bridges_and_filters_splits(tmp_path, tokenizer):
    rows = [{"name": "ok", "prompt": PROMPT, "target": "Paris", "intermediates": ["France"]},
            {"name": "bad", "prompt": PROMPT, "target": "Paris", "intermediates": ["Xqzvbnmplk"]},
            {"name": "other", "prompt": PROMPT, "target": "Berlin", "intermediates": ["Germany"]}]
    data = write(tmp_path / "d.json", json.dumps({"items": rows}))
    split_file = write(tmp_path / "s.json", json.dumps({"one": ["ok", "bad"]}))
    cfg = EvalConfig(datasets={"paper_multihop": data})
    everything = examples(tokenizer, "paper_multihop", ALL_SPLITS, cfg, split_file)
    assert [it.name for it in everything] == ["ok", "other"]
    assert [it.name for it in examples(tokenizer, "paper_multihop", "one", cfg, split_file)] == ["ok"]


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


def _save_lens(lm, path: Path, layers, scale: float = 0.0) -> Path:
    """A lens file at `path`: identity J matrices plus `scale` x seeded Gaussian noise."""
    generator = torch.Generator().manual_seed(7)
    d = lm.d_model
    jacobians = {l: torch.eye(d) + scale * torch.randn(d, d, generator=generator) for l in layers}
    jlens.JacobianLens(jacobians, n_prompts=2, d_model=d).save(str(path))
    return path


def test_attach_refit_reads_the_same_model_through_the_files_lens(qwen_lm, tmp_path):
    cfg = dataclasses.replace(JSpaceConfig(), readout_stats_dir=tmp_path / "stats", band_layers=(0, 1))
    js = load_jspace(qwen_lm, cfg, str(_save_lens(qwen_lm, tmp_path / "prefit.pt", range(3))))
    refit_file = _save_lens(qwen_lm, tmp_path / "refit.pt", range(2), scale=0.5)
    loaded = attach_refit(js, refit_file)
    own = loaded.own
    assert own.lm is js.lm and own.band == js.band and own.readout_stats is js.readout_stats
    assert own.lens_model is js.lens_model and js.lens.source_layers == [0, 1, 2]
    expected = jlens.JacobianLens.load(str(refit_file))
    assert own.lens.source_layers == [0, 1] and own.lens.n_prompts == 2
    for layer in (0, 1):
        assert torch.equal(own.lens.jacobians[layer].cpu(), expected.jacobians[layer])
    assert [r is x for r, x in zip(loaded.readers, [js, own], strict=True)] == [True, True]
    alone = attach_refit(js, None)
    assert alone.own is None and len(alone.readers) == 1 and alone.readers[0] is js


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
    assert result.answer_index == answer_index(qwen_lm.tokenizer, result.response_ids)
    assert result.shown_ids == result.response_ids[:result.answer_index + 1]


# ---- slice ----------------------------------------------------------------------------------

def test_slice_shapes_and_dtypes(qwen_lm):
    js = synthetic_js(qwen_lm)
    result = run_example(js, make_item(), "prompt", "none", 0, N_NEW)
    data = run_slice(js, result, top_n=5, max_tracked=7)
    seq, n_layers = len(result.prompt_ids) + len(result.shown_ids), len(data.layers)
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
    js.readout_stats = synthetic_stats(js)
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


def test_pinned_tokens_add_single_token_chinese_and_skip_multi_token(qwen_lm):
    js = synthetic_js(qwen_lm)
    tok = js.lm.tokenizer
    multi = tok("爱因斯坦", add_special_tokens=False).input_ids
    single = tok("法国", add_special_tokens=False).input_ids
    assert len(multi) > 1 and len(single) == 1
    result = run_example(js, make_item(bridge_zh=("法国", "爱因斯坦")), "prompt", "none", 0, N_NEW)
    pins = pinned_tokens(js, result)
    assert single[0] in pins and not set(multi) & set(pins)
    plain = run_example(js, make_item(bridge_zh=()), "prompt", "none", 0, N_NEW)
    old = set(token_variants(tok, "France")) | answer_token_ids(tok, plain.answers)
    assert pinned_tokens(js, plain) == sorted(old)


def test_bridge_ablation_changes_the_slice(qwen_lm):
    js = synthetic_js(qwen_lm)
    clean = run_example(js, make_item(), "prompt", "none", 0, N_NEW)
    ablated = run_example(js, make_item(), "prompt", "bridge", 0, N_NEW)
    a, b = run_slice(js, clean, 5, 6), run_slice(js, ablated, 5, 6)
    n = len(clean.prompt_ids)
    assert not np.array_equal(a.top_ids[:n], b.top_ids[:n])


def _assert_same_slice(a: jlens.vis.SliceData, b: jlens.vis.SliceData) -> None:
    assert a.layers == b.layers and a.context_token_ids == b.context_token_ids
    assert np.array_equal(a.top_ids, b.top_ids) and np.array_equal(a.top_ranks, b.top_ranks)
    assert a.tracked_token_ids == b.tracked_token_ids and np.array_equal(a.rank_tensor, b.rank_tensor)
    assert a.vocab_fragment == b.vocab_fragment and a.pinned_token_ids == b.pinned_token_ids


def _random_lens_js(lm, layers=(0, 1)):
    """synthetic_js on the same lm reading through seeded random J matrices (n_prompts=2)."""
    own = synthetic_js(lm, layers=layers)
    generator = torch.Generator().manual_seed(5)
    d = lm.d_model
    own.lens = jlens.JacobianLens({l: torch.randn(d, d, generator=generator) / d ** 0.5 for l in layers},
                                  n_prompts=2, d_model=d)
    return own


def test_run_slices_with_one_reader_equals_run_slice(qwen_lm):
    js = synthetic_js(qwen_lm)
    result = run_example(js, make_item(), "prompt", "bridge", 0, N_NEW)
    (only,) = run_slices(js, [js], result, 5, 6)
    _assert_same_slice(only, run_slice(js, result, 5, 6))


def test_run_slices_reads_a_second_lens_from_one_forward_pass(qwen_lm, monkeypatch):
    import viz.slice
    js, own = synthetic_js(qwen_lm, layers=(0, 1)), _random_lens_js(qwen_lm)
    result = run_example(js, make_item(), "prompt", "bridge", 0, N_NEW)
    calls = []

    def counted_forward(*args, **kwargs):
        calls.append(args[2])   # layers
        return forward_residuals(*args, **kwargs)
    monkeypatch.setattr(viz.slice, "forward_residuals", counted_forward)
    first, second = run_slices(js, [js, own], result, 5, 6)
    assert calls == [[0, 1, 2]]
    enc = shown_encoding(js, result)
    from_js = interventions_for_item(js, result.item, "bridge", 0)(enc)   # never built from own's lens
    _assert_same_slice(second, slice_from_ids(own, enc, from_js, 5, 6, pinned_tokens(js, result)))
    assert not np.array_equal(first.top_ids[:, :-1], second.top_ids[:, :-1])
    assert np.array_equal(first.top_ids[:, -1], second.top_ids[:, -1])


def test_replay_mismatches_reports_a_wrong_token(qwen_lm):
    js = synthetic_js(qwen_lm)
    result = run_example(js, make_item(), "prompt", "none", 0, N_NEW)
    data = run_slice(js, result, 3, 4)
    data.top_ids[len(result.prompt_ids) - 1, -1, 0] += 1
    assert replay_mismatches(data, result) == [0]
    # The last shown position's top-1 predicts a token that is not shown, so it is never checked.
    data.top_ids[len(result.prompt_ids) + len(result.shown_ids) - 1, -1, 0] += 1
    assert replay_mismatches(data, result) == [0]


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
    assert meta["T"] == len(result.prompt_ids) + len(result.shown_ids)
    assert meta["pinned"] == pinned_tokens(js, result)
    assert "bridge" in meta["what"] and meta["title"] == "tiny-qwen3"


def test_summary_mentions_response_and_mismatches(qwen_lm):
    result = run_example(synthetic_js(qwen_lm), make_item(), "prompt", "none", 0, N_NEW)
    n = len(result.prompt_ids)
    clean = summary(result, [])
    assert one_line(result.response_text) in clean and "reproduces generation" in clean
    assert one_line("a\nb") == "a⏎b"
    assert f"positions {n}-{n + result.answer_index}" in clean
    assert f"position {n + result.answer_index} is the answer token" in clean
    assert f"positions {n}" in summary(result, [0])
    assert "法国" in clean and "France (法国)" in clean


# ---- dictionary -----------------------------------------------------------------------------

def test_parse_line_reads_both_bracket_forms():
    assert parse_line("已經 已经 [yi3 jing1] /already/") == Entry("已經", "已经", "yi3 jing1", ("already",))
    assert parse_line("好多 好多 [[hao3duo1]] /many; quite a lot/much/\r\n") == Entry(
        "好多", "好多", "hao3duo1", ("many; quite a lot", "much"))


def test_parse_cedict_indexes_traditional_and_simplified():
    assert SAMPLE.index["已經"] == SAMPLE.index["已经"] == (parse_line(SAMPLE_LINES[6]),)
    assert [e.traditional for e in SAMPLE.index["后"]] == ["後"] and "後" in SAMPLE.index
    assert SAMPLE.n_entries == 13 and SAMPLE.max_len == 2


def test_parse_cedict_puts_proper_nouns_last_in_file_order():
    assert [e.pinyin for e in SAMPLE.index["都"]] == ["dou1", "du1", "Du1"]


def test_parse_cedict_skips_comments_blank_and_crlf_lines():
    cedict = parse_cedict(["# header\r\n", "\r\n", "  \n", "好 好 [hao3] /good/\r\n"])
    assert cedict.n_entries == 1 and cedict.index["好"][0].senses == ("good",)


def test_parse_cedict_raises_on_a_malformed_line_with_its_number():
    with pytest.raises(ValueError, match="line 3"):
        parse_cedict(["# c", "好 好 [hao3] /good/", "好 [hao3] good"])


def test_parse_cedict_raises_on_no_entries():
    with pytest.raises(ValueError, match="no CC-CEDICT entries"):
        parse_cedict(["# only a header", ""])


def _words(text: str) -> list[tuple[str, bool]]:
    return [(s.word, bool(s.entries)) for s in segment(SAMPLE, text)]


def test_segment_takes_the_longest_headword():
    assert _words("都已经") == [("都", True), ("已经", True)]
    assert _words(" 了好多") == [("了", True), ("好多", True)]
    assert segment(SAMPLE, "好多")[0] == Segment("好多", SAMPLE.index["好多"])


def test_segment_merges_unknown_runs_and_drops_whitespace():
    assert _words("我们好") == [("我们", False), ("好", True)]
    assert _words("我 们\n好 多") == [("我", False), ("们", False), ("好", True), ("多", True)]
    assert _words("�好") == [("�", False), ("好", True)]
    assert _words("abc好x") == [("abc", False), ("好", True), ("x", False)]
    assert segment(SAMPLE, "  ") == []


def test_load_cedict_reads_a_gzipped_file_with_bom(tmp_path):
    path = tmp_path / CEDICT_FILE
    with gzip.open(path, "wt", encoding="utf-8-sig", newline="\r\n") as f:
        f.write("\n".join(SAMPLE_LINES) + "\n")
    assert load_cedict(path) == SAMPLE


def test_load_cedict_missing_file_says_where_to_get_it(tmp_path):
    missing = tmp_path / "nowhere" / CEDICT_FILE
    with pytest.raises(FileNotFoundError) as info:
        load_cedict(missing)
    message = str(info.value)
    for part in (str(missing), CEDICT_PAGE, CEDICT_FILE, "download it in a browser", "dictionary_file: null"):
        assert part in message


# ---- glosses --------------------------------------------------------------------------------

def test_is_foreign_means_letters_beyond_latin():
    assert all(map(is_foreign, ["的", " 都已经", "こん", "Привет", "한국", "a的"]))
    assert not any(map(is_foreign, [" hello", "café", "Ångström", ",", "�", "123", "", " "]))


def test_token_gloss_strips_and_segments():
    assert token_gloss(SAMPLE, " 都已经") == {"text": "都已经", "segments": [
        {"word": "都", "entries": [{"reading": "dou1", "senses": ["all", "both"]},
                                  {"reading": "du1", "senses": ["capital city", "metropolis"]},
                                  {"reading": "Du1", "senses": ["surname Du"]}]},
        {"word": "已经", "entries": [{"reading": "yi3 jing1", "senses": ["already"]}]}]}


def _slice_data(ctx_offset: int = 0) -> jlens.vis.SliceData:
    """T=6 positions, 3 layers, top_n=3, over a vocab mixing Chinese, English, Japanese, punctuation."""
    vocab = {1: "的", 2: "都已经", 3: " hello", 4: "こん", 5: ",", 6: "�", 7: "café", 8: " 好多"}
    rng = np.random.default_rng(0)
    top_ids = rng.choice(np.array(list(vocab), dtype=np.int32), size=(6, 3, 3))
    tracked = sorted(vocab)
    return jlens.vis.SliceData(
        seq_len=6, layers=[0, 1, 2], context_token_ids=[3, 1, 2, 5, 4, 8],
        context_token_strs=[vocab[t] for t in [3, 1, 2, 5, 4, 8]], top_ids=top_ids,
        top_ranks=np.broadcast_to(np.arange(3, dtype=np.int32), (6, 3, 3)).copy(),
        tracked_token_ids=tracked, rank_tensor=rng.integers(0, 50, (6, 3, len(tracked)), dtype=np.int32),
        vocab_fragment=vocab, vocab_size=100, pinned_token_ids=[1], ctx_offset=ctx_offset)


def test_gloss_payload_keys_only_foreign_tokens():
    payload = gloss_payload(_slice_data(), SAMPLE)
    assert set(payload["glosses"]) == {"1", "2", "4", "8"}
    assert payload["context_ids"] == [3, 1, 2, 5, 4, 8]
    assert payload["glosses"]["4"] == {"text": "こん", "segments": [{"word": "こん", "entries": []}]}
    assert payload["glosses"]["8"]["text"] == "好多"
    assert gloss_payload(_slice_data(ctx_offset=2), SAMPLE)["context_ids"] == [2, 5, 4, 8]


def test_gloss_payload_on_a_real_slice(qwen_lm):
    js = synthetic_js(qwen_lm)
    result = run_example(js, make_item(), "prompt", "none", 0, N_NEW)
    data = run_slice(js, result, 5, 6)
    payload = gloss_payload(data, SAMPLE)
    assert payload["context_ids"] == list(result.prompt_ids + result.shown_ids)
    for tid, gloss in payload["glosses"].items():
        assert is_foreign(data.vocab_fragment[int(tid)]) and gloss["text"] == data.vocab_fragment[int(tid)].strip()
    assert json.loads(json.dumps(payload)) == payload


# ---- translation box in the page ------------------------------------------------------------

def _page(monkeypatch) -> str:
    monkeypatch.setitem(jlens.vis._TEMPLATE_FOR_MODE, "embed",
                        jlens.vis.PAGE_TEMPLATE.replace("__D3__", "<script>/*d3*/</script>"))
    return jlens.vis.build_page(_slice_data(), "的都已经", title="t", description="d", mode="embed")[0]


def _gloss_data(page: str) -> dict:
    start = page.index(f'id="{GLOSS_DATA_ID}">') + len(f'id="{GLOSS_DATA_ID}">')
    return json.loads(page[start:page.index("</script>", start)])


def test_with_glosses_keeps_the_page_and_embeds_the_payload(monkeypatch):
    page = _page(monkeypatch)
    payload = gloss_payload(_slice_data(), SAMPLE)
    glossed = with_glosses(page, payload)
    assert _bootstrap(glossed) == _bootstrap(page)
    assert _gloss_data(glossed) == payload
    assert glossed.index(GLOSS_DATA_ID) < glossed.rindex("</body>")
    assert glossed[glossed.rindex("</body>"):] == page[page.rindex("</body>"):]
    assert glossed.count("</script>") == page.count("</script>") + 2


def test_with_glosses_dictionary_text_cannot_close_the_script(monkeypatch):
    page = _page(monkeypatch)
    payload = {"context_ids": [], "glosses": {"1": {"text": "的", "segments": [{"word": "的", "entries": [
        {"reading": "de5", "senses": ["</script><script>alert(1)</script>", "<!--<script>"]}]}]}}}
    glossed = with_glosses(page, payload)
    assert glossed.count("</script>") == page.count("</script>") + 2
    assert _gloss_data(glossed) == payload


def test_with_glosses_round_trips_through_srcdoc(monkeypatch):
    glossed = with_glosses(_page(monkeypatch), gloss_payload(_slice_data(), SAMPLE))
    srcdoc = re.search(r'srcdoc="([^"]*)"', iframe(glossed, 100)).group(1)
    assert html.unescape(srcdoc) == glossed


def test_with_glosses_needs_a_body():
    with pytest.raises(ValueError, match="</body>"):
        with_glosses("<html>no body</html>", {"context_ids": [], "glosses": {}})


def _static(name: str) -> str:
    return (Path(__file__).resolve().parent.parent / "viz" / "static" / name).read_text(encoding="utf-8")


def test_static_files_cannot_close_their_tags():
    for name in ("gloss.js", "gloss.css"):
        assert "</script" not in _static(name).lower() and "</style" not in _static(name).lower()
    assert f"'{GLOSS_DATA_ID}'" in _static("gloss.js")


# What gloss.js relies on in the jlens page (left: as gloss.js uses it; right: the template's
# code that makes it true). A jlens upgrade that changes any of these fails here.
JLENS_CONTRACT = [
    (".jlens-page", '<div class="jlens-page">'),
    (".token-cell", "c.className = 'token-cell'"),
    (".pinned-chip", 'class="pinned-chip'),
    ("dataset.tid", 'data-tid="${tid}"'),
    ("dataset.tid", "c.dataset.tid = tid"),
    (".chip-x", '<span class="chip-x" data-tid="${tid}"'),
    ("#tokens span.token", 'id="tokens"'),
    ("#tokens span.token", "sp.className = 'token'; sp.textContent = s;"),
    ("context_ids", "meta.ctx_strs = meta.ctx_strs.slice(meta.ctx_offset)"),
    ("#hm-wrap", 'id="hm-wrap"'),
    (".hm-hover", "hmHover.style.display = 'block'"),
    (".hm-hover", "if (!c) { hmHover.style.display = 'none'"),
    ("#byLayer .token-row.active-row .token-cell", 'id="byLayer"'),
    ("#byLayer .token-row.active-row .token-cell", "row.classList.add('active-row')"),
    ("#byLayer .token-row.active-row .token-cell", "fillRow(byLayerRows[i].cells, selCtx, L - 1 - i)"),
    ("#byLayer .token-row.active-row .token-cell", "const tid = at(D.top_ids, t, li, k), c = cells[k];"),
    ("#byLayer .token-row.active-row .token-cell", "argmax[t][l] = at(D.top_ids, t, l, 0)"),
    ("#hm-wrap", "selCtx = c.t; selLayer = layers[c.li]; pinToken(argmax[c.t][c.li]);"),
    ("#hm-wrap", "refillPanels(); render();\n}"),
    ("--jb-gloss-z:900", ".tooltip{position:fixed"),
    ("--jb-gloss-z:900", "z-index:1000"),
]


@pytest.mark.parametrize("ours, theirs", JLENS_CONTRACT)
def test_jlens_page_contract(ours, theirs):
    assert ours in _static("gloss.js") + _static("gloss.css")
    assert theirs in jlens.vis.PAGE_TEMPLATE


# ---- app (gradio) ---------------------------------------------------------------------------

DATASET = "paper_multihop"
TINY_CHOICES = {"tiny": ModelChoice("tiny"), "tiny + a": ModelChoice("tiny", Path("a"))}


def _app_context(js, tmp_path, dictionary=None, load=None, own=None):
    """An AppContext over `js` (with `own` as the refit-lens reader); `load` (choice -> Loaded)
    replaces the session's loader."""
    from viz.app import AppContext
    rows = [{"name": "ok", "prompt": PROMPT, "target": "Paris", "intermediates": ["France"]}]
    cfg = EvalConfig(datasets={DATASET: write(tmp_path / "d.json", json.dumps({"items": rows}))})
    viz = dataclasses.replace(load_viz_config(DEFAULT_VIZ_CONFIG), split_file=None, top_n=3, max_tracked=4,
                              lenses_dir=tmp_path / "lenses")
    return AppContext(run=load_run_config(DEFAULT_RUN_CONFIG), viz=viz, eval_cfg=cfg,
                      session=Session(load or (lambda choice: Loaded(js, own))), tokenizer=js.lm.tokenizer,
                      has_stats=js.readout_stats is not None, choices=TINY_CHOICES,
                      default_model="tiny", rng=random.Random(0), dictionary=dictionary, cache=ResultCache())


def _offline_pages(monkeypatch) -> None:
    """Seed jlens's embed-template memo so building a page never fetches d3."""
    monkeypatch.setitem(jlens.vis._TEMPLATE_FOR_MODE, "embed",
                        jlens.vis.PAGE_TEMPLATE.replace("__D3__", "<script>/*d3*/</script>"))


def _values(dropdown) -> list[str]:
    return [value for _, value in dropdown.choices]


def test_app_builds_and_choice_handlers_filter_invalid_options(qwen_lm, tmp_path):
    gr = pytest.importorskip("gradio")
    from viz.app import build_app, on_dataset, on_example, on_split
    ctx = _app_context(synthetic_js(qwen_lm), tmp_path)
    assert isinstance(build_app(ctx), gr.Blocks)
    assert _values(on_dataset(ctx, "paper_multihop")) == [ALL_SPLITS]
    items, example = on_split(ctx, "paper_multihop", ALL_SPLITS)
    assert list(items) == ["ok"] and example.value == "ok"
    no_swap = {"t": make_item(swap_to=None)}
    fields, conditions = on_example(ctx, no_swap, "t", "hop2_prompt", "wrong_bridge")
    assert _values(fields) == ["prompt", "hop1_prompt"] and fields.value == "prompt"
    assert "wrong_bridge" not in _values(conditions) and conditions.value == "none"
    assert not NEEDS_STATS & set(_values(conditions))   # synthetic_js has no readout stats


def test_profile_choices_keep_the_session_profile_and_its_adapters(tmp_path):
    pytest.importorskip("gradio")
    from viz.run_app import profile_choices
    runs = tmp_path / "runs"
    _adapter_dir(runs, "mexp", "qwen3-1.7b")
    run = load_run_config(DEFAULT_RUN_CONFIG)
    viz = dataclasses.replace(load_viz_config(DEFAULT_VIZ_CONFIG), runs_dir=runs)
    assert list(profile_choices(run, viz, "qwen3-1.7b")) == ["qwen3-1.7b", "qwen3-1.7b + mexp"]
    with pytest.raises(ValueError, match="no prefit lens"):
        profile_choices(run, viz, "qwen3-0.6b")


def test_every_cascade_ends_in_show_and_only_show_and_run_use_the_model_queue(qwen_lm, tmp_path):
    pytest.importorskip("gradio")
    from viz.app import build_app
    demo = build_app(_app_context(synthetic_js(qwen_lm), tmp_path))
    deps = demo.config["dependencies"]
    parents = {d["trigger_after"] for d in deps}
    leaves = sorted(d["api_name"] for d in deps if d["id"] not in parents)
    assert leaves[0] == "run" and leaves[1:] and all(name.startswith("show") for name in leaves[1:])
    queued = {fn.api_name for fn in demo.fns.values() if fn.concurrency_id == "model"}
    names = [d["api_name"] for d in deps]
    assert queued == {name for name in names if name == "run" or name.startswith("show")}


def test_app_run_handler_returns_summary_and_iframe(qwen_lm, tmp_path, monkeypatch):
    gr = pytest.importorskip("gradio")
    from viz.app import on_run
    _offline_pages(monkeypatch)
    ctx = _app_context(synthetic_js(qwen_lm), tmp_path)
    items = {"t": make_item()}
    _, text, embed, own = on_run(ctx, "tiny", DATASET, items, "t", "prompt", "bridge")
    assert text.startswith("**Model:** tiny\n") and "**Response** (bridge)" in text
    assert embed.startswith("<iframe") and "tiny: t | original lens (1 prompts)" in html.unescape(embed)
    assert "No refit lens" in own
    assert html.escape(f"python -m jspace.fit_lens --model tiny --out {tmp_path / 'lenses' / 'tiny.pt'}") in own
    with pytest.raises(gr.Error):
        on_run(ctx, "tiny", DATASET, items, None, "prompt", "none")


def test_app_run_embeds_glosses_only_with_a_dictionary(qwen_lm, tmp_path, monkeypatch):
    pytest.importorskip("gradio")
    from viz.app import on_run
    _offline_pages(monkeypatch)
    js, items = synthetic_js(qwen_lm), {"t": make_item()}
    with_dict = on_run(_app_context(js, tmp_path, SAMPLE), "tiny", DATASET, items, "t", "prompt", "none")[2]
    without = on_run(_app_context(js, tmp_path, None), "tiny", DATASET, items, "t", "prompt", "none")[2]
    assert f'id="{GLOSS_DATA_ID}"' in html.unescape(with_dict)
    assert GLOSS_DATA_ID not in html.unescape(without)


def test_app_refit_pane_shows_the_refit_lens_page_when_the_model_has_one(qwen_lm, tmp_path, monkeypatch):
    pytest.importorskip("gradio")
    from viz.app import on_run
    _offline_pages(monkeypatch)
    js, own = synthetic_js(qwen_lm, layers=(0, 1)), _random_lens_js(qwen_lm)
    ctx = _app_context(js, tmp_path, SAMPLE, own=own)
    _, _, original, refit = on_run(ctx, "tiny", DATASET, {"t": make_item()}, "t", "prompt", "bridge")
    assert refit.startswith("<iframe") and "tiny: t | refit lens (2 prompts)" in html.unescape(refit)
    assert "tiny: t | original lens (1 prompts)" in html.unescape(original)
    assert f'id="{GLOSS_DATA_ID}"' in html.unescape(refit) and refit != original


def _counting_app(js, tmp_path, monkeypatch):
    """An AppContext whose loader records each load and whose generation (viz.app.run_example)
    records each call; returns (ctx, loads, generations)."""
    import viz.app
    loads, generations = [], []

    def load(choice):
        loads.append(choice)
        return Loaded(js)

    def counted_run_example(*args, **kwargs):
        generations.append(args[2:4])   # (field, condition)
        return run_example(*args, **kwargs)
    monkeypatch.setattr(viz.app, "run_example", counted_run_example)
    _offline_pages(monkeypatch)
    return _app_context(js, tmp_path, load=load), loads, generations


def test_app_show_and_run_load_and_generate_once_per_model_and_question(qwen_lm, tmp_path, monkeypatch):
    pytest.importorskip("gradio")
    from viz.app import on_run, on_show
    ctx, loads, generations = _counting_app(synthetic_js(qwen_lm), tmp_path, monkeypatch)
    items = {"t": make_item(), "u": make_item(name="u")}
    placeholder = ("", "Not run yet for **tiny** on this question. Press Run.", "", "")   # no table yet
    assert on_show(ctx, "tiny", DATASET, "t", "prompt", "none") == placeholder and loads == []
    first = on_run(ctx, "tiny", DATASET, items, "t", "prompt", "none")
    assert on_run(ctx, "tiny", DATASET, items, "t", "prompt", "none") == first
    assert len(loads) == 1 and len(generations) == 1
    other = on_run(ctx, "tiny + a", DATASET, items, "t", "prompt", "none")
    assert other[1].startswith("**Model:** tiny + a\n") and len(loads) == 2 and len(generations) == 2
    back = on_show(ctx, "tiny", DATASET, "t", "prompt", "none")
    assert back[1:] == first[1:] and back[0] == other[0]   # the table now lists both runs
    assert len(loads) == 2 and len(generations) == 2
    on_run(ctx, "tiny", DATASET, items, "u", "prompt", "none")
    assert on_show(ctx, "tiny", DATASET, "t", "prompt", "none") == placeholder


def test_app_table_lists_the_questions_runs_in_run_order(qwen_lm, tmp_path, monkeypatch):
    pytest.importorskip("gradio")
    from viz.app import on_run, on_show
    ctx, _, _ = _counting_app(synthetic_js(qwen_lm), tmp_path, monkeypatch)
    items = {"t": make_item(), "u": make_item(name="u")}
    one = on_run(ctx, "tiny + a", DATASET, items, "t", "prompt", "none")[0]
    table = on_run(ctx, "tiny", DATASET, items, "t", "prompt", "bridge")[0]
    rows = table.splitlines()[2:]
    assert len(one.splitlines()) == 3 and len(rows) == 2
    assert rows[0].startswith("| tiny + a | prompt | none |") and rows[1].startswith("| tiny | prompt | bridge |")
    assert on_show(ctx, "tiny", DATASET, "u", "prompt", "none")[0] == ""


# ---- cache ----------------------------------------------------------------------------------

def _shown(text: str) -> Shown:
    return Shown(summary=text, original=f"<iframe>{text}</iframe>", own="", response=text, correct=False)


def test_result_cache_round_trip_and_two_models_on_one_question():
    cache = ResultCache()
    a, b = RunKey("d", "t", "prompt", "none", "m1"), RunKey("d", "t", "prompt", "none", "m2")
    assert cache.get(a) is None
    cache.put(a, _shown("a"))
    cache.put(b, _shown("b"))
    assert cache.get(a) == _shown("a") and cache.get(b) == _shown("b")
    assert a.question == b.question == ("d", "t")


def test_result_cache_drops_the_old_question_and_lists_entries_in_order():
    cache = ResultCache()
    keys = [RunKey("d", "t", "prompt", c, m) for c, m in [("none", "m1"), ("bridge", "m1"), ("none", "m2")]]
    for i, key in enumerate(keys):
        cache.put(key, _shown(str(i)))
    assert cache.entries(("d", "t")) == [(k, _shown(str(i))) for i, k in enumerate(keys)]
    new = RunKey("d", "u", "prompt", "none", "m1")
    cache.put(new, _shown("new"))
    assert [cache.get(k) for k in keys] == [None, None, None]
    assert cache.entries(("d", "t")) == [] and cache.entries(("d", "u")) == [(new, _shown("new"))]


def test_responses_table_rows_and_escaping():
    assert responses_table([]) == ""
    rows = [(RunKey("d", "t", "prompt", "none", "m1"), _shown("Paris")),
            (RunKey("d", "t", "hop1_prompt", "bridge", "m | 2"),
             dataclasses.replace(_shown("a|b\nc <i>x</i>"), correct=True))]
    lines = responses_table(rows).splitlines()
    assert lines[0] == "| Model | Prompt field | Ablation | Response | Correct |" and len(lines) == 4
    assert lines[2] == "| m1 | prompt | none | Paris | no |"
    assert lines[3] == "| m \\| 2 | hop1_prompt | bridge | a\\|b⏎c &lt;i&gt;x&lt;/i&gt; | yes |"


def test_no_refit_lens_note_escapes_and_shows_the_command():
    note = no_refit_lens_note("m <1>", "python -m jspace.fit_lens --out a&b.pt")
    assert "m &lt;1&gt;" in note and "<pre>python -m jspace.fit_lens --out a&amp;b.pt</pre>" in note
    assert "restart the viewer" in note


def test_item_card_shows_the_fields_prompt_and_answers():
    card = item_card(make_item(), "hop1_prompt")
    assert "France's neighbour is" in card and "**Expected:** France" in card and "Germany" in card
    assert "France (法国)" in card
    assert "**Bridge:** France |" in item_card(make_item(bridge_zh=()), "hop1_prompt")
