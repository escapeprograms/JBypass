# tests/model

CPU unit tests for `jbypass/model` on tiny randomly initialised models of two architectures
(Qwen3-shaped and GPT-2-shaped), each described only by a `ModelProfile`. They use the real
Qwen3 tokenizer (small cached download); no model weights are downloaded. Run: `python -m pytest tests/model`.

- `conftest.py` — profiles, settings and builders (`tiny_qwen3`, `tiny_gpt2`, `build_qwen`, `build_gpt2`); fixtures `tokenizer`, `lm` (runs a test once per architecture), `qwen_lm`; `random_ids` for raw forward passes.
- `dummies.py` — test-only interventions: `Identity` (records its input), `ZeroOut`, `Replace` (applies a function once the sequence reaches a length).
- `test_config_loading.py` — YAML loading, profile-name check, sizes read from the model, left padding, `token_ids`, and the post-norm `hidden_states` fact.
- `test_hooks.py` — identity vs zero-out, hook removal on exit and exception, capture shapes, `unembed(last capture) == logits`, post-intervention capture, gradients through interventions, tensor/tuple block-output normalisation.
- `test_inference.py` — chat template and kwargs, left padding, batched equals single (logits, loss, generation), `generate` returns only new text and stops at EOS.
- `test_lora_training.py` — collate text and labels, batch cycling, hooks still fire after LoRA, fp32 adapter over bf16 base, `adapter_disabled`, save/load round trip, wrong-profile load raises, only adapters change, loss falls, factory called once per batch, interventions sit in the training graph.
