import argparse
import dataclasses
from pathlib import Path

import pytest
import torch

from jspace.config import JSpaceConfig, jspace_config
from model.cli import add_profile_args, resolve_profile
from model.config import LensSource, load_model_profile, load_run_config

from conftest import D_MODEL, N_LAYERS, VOCAB, random_ids

RUN_YAML = Path(__file__).resolve().parents[1] / "configs" / "run.yaml"
EXP3_YAML = RUN_YAML.parent / "exp3.yaml"


def test_run_config_loads_and_points_at_profiles():
    config = load_run_config(RUN_YAML)
    profile = load_model_profile(config.model, config.profiles_dir)
    assert profile.name == config.model
    assert config.generation.max_new_tokens > 0


@pytest.mark.parametrize("name", ["qwen3-1.7b", "qwen3-0.6b"])
def test_every_profile_loads(name):
    profile = load_model_profile(name, load_run_config(RUN_YAML).profiles_dir)
    assert profile.name == name
    assert profile.chat_template_kwargs == {"enable_thinking": False}


def test_prefit_lens_is_parsed_per_profile():
    profiles_dir = load_run_config(RUN_YAML).profiles_dir
    lens = load_model_profile("qwen3-1.7b", profiles_dir).prefit_lens
    assert isinstance(lens, LensSource) and lens.repo == "neuronpedia/jacobian-lens"
    assert lens.file.endswith("Qwen3-1.7B_jacobian_lens.pt") and lens.revision == "main"
    assert load_model_profile("qwen3-0.6b", profiles_dir).prefit_lens is None


def test_device_flag_overrides_profile_only_when_given():
    parser = argparse.ArgumentParser()
    add_profile_args(parser)
    _, default = resolve_profile(parser.parse_args(["--config", str(RUN_YAML)]))
    _, cpu = resolve_profile(parser.parse_args(["--config", str(RUN_YAML), "--device", "cpu"]))
    assert default.device == "cuda" and cpu.device == "cpu"


def test_profile_name_must_match_file(tmp_path):
    (tmp_path / "a.yaml").write_text(
        "name: b\nhf_id: x\ndtype: float32\ndevice: cpu\nblocks_path: x\nfinal_norm_path: x\n"
        "chat_template_kwargs: {}\nlora_target_modules: []\nprefit_lens: null\n"
    )
    with pytest.raises(ValueError):
        load_model_profile("a", tmp_path)


def _run_yaml_with(tmp_path, jspace_section: str = "") -> Path:
    """A copy of configs/run.yaml in tmp_path/cfg/ with `jspace_section` appended."""
    folder = tmp_path / "cfg"
    folder.mkdir()
    path = folder / "run.yaml"
    path.write_text(RUN_YAML.read_text(encoding="utf-8") + jspace_section, encoding="utf-8")
    return path


def test_run_config_keeps_raw_jspace_section_and_config_dir(tmp_path):
    path = _run_yaml_with(tmp_path, "jspace:\n  clamp_z: 1.5\n")
    run = load_run_config(path)
    assert run.jspace == {"clamp_z": 1.5}
    assert run.config_dir == path.parent.resolve()
    assert load_run_config(RUN_YAML).config_dir == RUN_YAML.parent


def test_no_jspace_section_gives_the_default_jspace_config():
    run = load_run_config(RUN_YAML)
    assert run.jspace == {}
    assert jspace_config(run) == JSpaceConfig()


def test_empty_jspace_section_gives_the_default_jspace_config(tmp_path):
    assert jspace_config(load_run_config(_run_yaml_with(tmp_path, "jspace:\n"))) == JSpaceConfig()


def test_jspace_override_is_applied(tmp_path):
    run = load_run_config(_run_yaml_with(tmp_path, "jspace:\n  clamp_z: 1.5\n  ablation_top_k: 3\n"))
    assert jspace_config(run) == dataclasses.replace(JSpaceConfig(), clamp_z=1.5, ablation_top_k=3)


def test_unknown_jspace_key_raises_naming_it(tmp_path):
    run = load_run_config(_run_yaml_with(tmp_path, "jspace:\n  clamp_zz: 1.5\n"))
    with pytest.raises(ValueError, match="clamp_zz"):
        jspace_config(run)


def test_readout_stats_dir_is_relative_to_the_config_folder(tmp_path):
    run = load_run_config(_run_yaml_with(tmp_path, "jspace:\n  readout_stats_dir: ../stats/exp\n"))
    assert jspace_config(run).readout_stats_dir == (tmp_path / "stats" / "exp").resolve()


def test_band_layers_list_becomes_a_tuple(tmp_path):
    run = load_run_config(_run_yaml_with(tmp_path, "jspace:\n  band_layers: [8, 26]\n"))
    assert jspace_config(run).band_layers == (8, 26)


@pytest.mark.parametrize("value", ["[8]", "[8, 26, 27]", "[8.0, 26]", "8", "[true, 26]", "'8-26'"])
def test_band_layers_must_be_two_ints(tmp_path, value):
    run = load_run_config(_run_yaml_with(tmp_path, f"jspace:\n  band_layers: {value}\n"))
    with pytest.raises(ValueError, match="band_layers"):
        jspace_config(run)


def test_fit_checkpoint_every_defaults_to_10():
    assert JSpaceConfig().fit_checkpoint_every == 10


@pytest.mark.parametrize("value, expected", [("null", None), ("25", 25)])
def test_fit_checkpoint_every_from_yaml(tmp_path, value, expected):
    run = load_run_config(_run_yaml_with(tmp_path, f"jspace:\n  fit_checkpoint_every: {value}\n"))
    assert jspace_config(run).fit_checkpoint_every == expected


def test_exp3_config_differs_from_run_yaml_only_in_its_jspace_section():
    run, exp3 = load_run_config(RUN_YAML), load_run_config(EXP3_YAML)
    cfg = jspace_config(exp3)
    assert cfg.band_layers == (8, 26)
    assert cfg.readout_stats_dir == (RUN_YAML.parents[1] / "stats" / "exp3").resolve()
    assert cfg == dataclasses.replace(JSpaceConfig(), band_layers=(8, 26),
                                      readout_stats_dir=cfg.readout_stats_dir)
    assert dataclasses.replace(exp3, jspace={}) == run


def test_default_readout_stats_dir_stays_cwd_relative():
    assert jspace_config(load_run_config(RUN_YAML)).readout_stats_dir == Path("stats")


def test_sizes_are_read_from_model(lm):
    assert lm.n_layers == N_LAYERS
    assert lm.d_model == D_MODEL
    assert lm.unembedding_matrix.shape == (VOCAB, D_MODEL)
    assert lm.dtype == torch.float32
    assert lm.device.type == "cpu"


def test_left_padding_is_set(lm):
    assert lm.tokenizer.padding_side == "left"


def test_token_ids_have_no_special_tokens(lm):
    assert lm.token_ids("hello") == lm.tokenizer("hello", add_special_tokens=False).input_ids


def test_unembed_matches_lm_head_on_final_hidden_state(lm):
    ids, mask = random_ids(lm)
    out = lm.hf_model(input_ids=ids, attention_mask=mask, output_hidden_states=True)
    # Only to check `unembed`; the module itself never relies on output_hidden_states.
    # In transformers 5.x the last hidden state here is already post-norm, so apply lm_head only.
    head = lm.hf_model.get_output_embeddings()
    torch.testing.assert_close(head(out.hidden_states[-1]), out.logits)
