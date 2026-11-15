import argparse
from pathlib import Path

import pytest
import torch

from model.cli import add_profile_args, resolve_profile
from model.config import LensSource, load_model_profile, load_run_config

from conftest import D_MODEL, N_LAYERS, VOCAB, random_ids

RUN_YAML = Path(__file__).resolve().parents[1] / "configs" / "run.yaml"


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
