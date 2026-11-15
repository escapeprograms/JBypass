"""End-to-end check of the model module on a real model (GPU).

Loads the profile's model, generates an answer, reruns it under a zero-out intervention,
captures residuals at every layer, runs a few LoRA steps with a dummy intervention factory,
and confirms adapter_disabled gives the pre-training logits. Reports losses and peak VRAM.
Raises on the first failed check.

Usage:  python scripts/smoke_test.py [--model qwen3-0.6b] [--config configs/run.yaml]
"""

from dataclasses import dataclass

import torch

from _cli import parse_run_args, resolve_run
from jbypass.model.config import RunConfig
from jbypass.model.data import Example
from jbypass.model.hooks import HookContext, capture_residuals
from jbypass.model.inference import encode, forward_logits, generate, next_token_logits
from jbypass.model.loading import LanguageModel, load_model
from jbypass.model.lora import adapter_disabled, attach_lora
from jbypass.model.prompts import chat_prompt
from jbypass.model.training import train

QUESTION = "How many legs does the animal that spins webs have?"
TOY_EXAMPLES = [
    Example("How many legs does the animal that spins webs have?", "Eight", bridge="spider"),
    Example("What colour is the sky on a clear day?", "Blue"),
    Example("What is the capital of the country whose flag is a red circle on white?", "Tokyo", bridge="Japan"),
]
MAX_LOGIT_DIFF = 1e-3   # adapter_disabled must reproduce base logits (bf16 noise only)


@dataclass
class ZeroOut:
    """Dummy intervention: replaces the residual with zeros."""

    layers: list[int]

    def __call__(self, resid: torch.Tensor, ctx: HookContext) -> torch.Tensor:
        return torch.zeros_like(resid)


@dataclass
class Identity:
    """Dummy intervention: passes the residual through, keeping it in the graph."""

    layers: list[int]

    def __call__(self, resid: torch.Tensor, ctx: HookContext) -> torch.Tensor:
        return resid * 1


def check(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(f"smoke test failed: {message}")


def check_generation(lm: LanguageModel, prompt: str, config: RunConfig) -> str:
    answer = generate(lm, [prompt], config.generation)[0]
    zeroed = generate(lm, [prompt], config.generation, [ZeroOut([lm.n_layers // 2])])[0]
    print(f"answer:           {answer!r}")
    print(f"zero-out layer {lm.n_layers // 2}: {zeroed!r}")
    check(zeroed != answer, "zero-out intervention did not change the output")
    return answer


def check_capture(lm: LanguageModel, prompt: str) -> None:
    enc = encode(lm, [prompt])
    with torch.no_grad(), capture_residuals(lm, range(lm.n_layers)) as resids:
        logits = forward_logits(lm, enc.input_ids, enc.attention_mask)
    expected = (1, enc.input_ids.shape[1], lm.d_model)
    check(len(resids) == lm.n_layers, "not every layer was captured")
    check(all(r.shape == expected for r in resids.values()), f"residual shapes differ from {expected}")
    diff = (lm.unembed(resids[lm.n_layers - 1]) - logits).abs().max().item()
    print(f"captured {lm.n_layers} layers of shape {expected}; unembed(last) vs logits max diff {diff:.2e}")


def check_lora(lm: LanguageModel, prompt: str, config: RunConfig) -> list[float]:
    base_logits = next_token_logits(lm, [prompt])
    attach_lora(lm, config.lora)
    factory_layer = lm.n_layers // 2
    losses = train(lm, TOY_EXAMPLES, config.train, chat_prompt, lambda batch: [Identity([factory_layer])])
    print(f"LoRA losses ({config.train.steps} steps): {[round(x, 4) for x in losses]}")
    print(f"answer after LoRA: {generate(lm, [prompt], config.generation)[0]!r}")
    with adapter_disabled(lm):
        diff = (next_token_logits(lm, [prompt]) - base_logits).abs().max().item()
    print(f"adapter_disabled vs pre-training logits max diff: {diff:.2e}")
    check(diff <= MAX_LOGIT_DIFF, "adapter_disabled does not reproduce the base logits")
    return losses


def main() -> None:
    config, profile = resolve_run(parse_run_args(__doc__))
    torch.cuda.reset_peak_memory_stats()
    lm = load_model(profile)
    print(f"model {profile.name} ({profile.hf_id}): {lm.n_layers} layers, d_model {lm.d_model}, {lm.dtype}")
    prompt = chat_prompt(lm, QUESTION)
    check_generation(lm, prompt, config)
    check_capture(lm, prompt)
    check_lora(lm, prompt, config)
    print(f"peak VRAM: {torch.cuda.max_memory_allocated() / 2**30:.2f} GiB")
    print("smoke test passed")


if __name__ == "__main__":
    main()
