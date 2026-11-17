"""Tiny randomly initialised models of two architectures, each described only by a ModelProfile.

Both use the real Qwen3 tokenizer (small download, cached) with a matching vocab size,
so no model weights are downloaded. Everything runs on CPU in fp32 unless a test says otherwise.
"""

import types

import jlens
import pytest
import torch
from transformers import AutoTokenizer, GPT2Config, GPT2LMHeadModel, Qwen3Config, Qwen3ForCausalLM

from jspace.baseline import ReadoutStats
from jspace.config import JSpaceConfig
from jspace.loading import lens_view
from model.config import GenerationSettings, LoraSettings, ModelProfile, TrainSettings
from model.loading import LanguageModel

TOKENIZER_ID = "Qwen/Qwen3-1.7B"
VOCAB = 151936
D_MODEL = 64
N_LAYERS = 3

QWEN_PROFILE = ModelProfile(
    name="tiny-qwen3", hf_id="none", dtype="float32", device="cpu",
    blocks_path="model.layers", final_norm_path="model.norm",
    chat_template_kwargs={"enable_thinking": False},
    lora_target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    prefit_lens=None,
)
GPT2_PROFILE = ModelProfile(
    name="tiny-gpt2", hf_id="none", dtype="float32", device="cpu",
    blocks_path="transformer.h", final_norm_path="transformer.ln_f",
    chat_template_kwargs={},
    lora_target_modules=["c_attn"],
    prefit_lens=None,
)

GENERATION = GenerationSettings(max_new_tokens=4)
LORA = LoraSettings(r=8, alpha=16, dropout=0.0)
TRAIN = TrainSettings(learning_rate=1e-2, batch_size=2, steps=3, seed=0)


@pytest.fixture(scope="session")
def tokenizer():
    return AutoTokenizer.from_pretrained(TOKENIZER_ID)


def tiny_qwen3() -> Qwen3ForCausalLM:
    torch.manual_seed(0)
    config = Qwen3Config(
        vocab_size=VOCAB, hidden_size=D_MODEL, intermediate_size=128, num_hidden_layers=N_LAYERS,
        num_attention_heads=4, num_key_value_heads=2, head_dim=16, max_position_embeddings=256,
        tie_word_embeddings=False,
    )
    return Qwen3ForCausalLM(config).eval()


def tiny_gpt2() -> GPT2LMHeadModel:
    torch.manual_seed(0)
    config = GPT2Config(vocab_size=VOCAB, n_positions=256, n_embd=D_MODEL, n_layer=N_LAYERS, n_head=4)
    return GPT2LMHeadModel(config).eval()


def build_qwen(tokenizer) -> LanguageModel:
    return LanguageModel(tiny_qwen3(), tokenizer, QWEN_PROFILE)


def build_gpt2(tokenizer) -> LanguageModel:
    return LanguageModel(tiny_gpt2(), tokenizer, GPT2_PROFILE)


@pytest.fixture(params=["qwen3", "gpt2"])
def lm(request, tokenizer) -> LanguageModel:
    """A fresh tiny LanguageModel; tests using it run once per architecture."""
    builders = {"qwen3": build_qwen, "gpt2": build_gpt2}
    return builders[request.param](tokenizer)


@pytest.fixture
def qwen_lm(tokenizer) -> LanguageModel:
    return build_qwen(tokenizer)


def random_ids(lm: LanguageModel, batch: int = 2, seq: int = 7) -> tuple[torch.Tensor, torch.Tensor]:
    """Random in-vocab token ids and an all-ones attention mask."""
    generator = torch.Generator().manual_seed(1)
    ids = torch.randint(0, 1000, (batch, seq), generator=generator)
    return ids, torch.ones_like(ids)


def synthetic_js(lm: LanguageModel, layers=(0, 1, 2), band=(1,)) -> types.SimpleNamespace:
    """A JSpace stand-in on a tiny model: a real JacobianLens with identity J matrices,
    so each token's J-lens vector is its unembedding row."""
    lens = jlens.JacobianLens({l: torch.eye(lm.d_model) for l in layers}, n_prompts=1, d_model=lm.d_model)
    return types.SimpleNamespace(
        lm=lm, lens=lens, lens_model=lens_view(lm), band=list(band), cfg=JSpaceConfig(),
        readout_stats=None,
    )


def synthetic_stats(js, seed: int = 0) -> ReadoutStats:
    """Small ReadoutStats for js's band layers: random mean, positive std, every token
    counted once, named after js.lm's profile."""
    generator = torch.Generator().manual_seed(seed)
    vocab = js.lm.unembedding_matrix.shape[0]
    shape = (len(js.band), vocab)
    return ReadoutStats(
        layers=list(js.band), mean=torch.randn(shape, generator=generator),
        std=torch.rand(shape, generator=generator) + 0.1, count=torch.ones(vocab, dtype=torch.long),
        model=js.lm.profile.name, n_texts=1,
    )