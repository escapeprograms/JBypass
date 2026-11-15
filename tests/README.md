# tests

CPU unit tests on tiny randomly initialised models of two architectures (Qwen3-shaped and
GPT-2-shaped), each described only by a `ModelProfile`. They use the real Qwen3 tokenizer
(small cached download); no model weights are downloaded. Run: `python -m pytest tests`.
Files are flat (no `tests/model/` folder, which could shadow the `model` package on import).

- `conftest.py` — profiles, settings and builders (`tiny_qwen3`, `tiny_gpt2`, `build_qwen`, `build_gpt2`); fixtures `tokenizer`, `lm` (runs a test once per architecture), `qwen_lm`; `random_ids` for raw forward passes; `synthetic_js(lm)`: a JSpace stand-in with a real `jlens.JacobianLens` of identity matrices (lens vectors = unembedding rows), band `[1]`.
- `dummies.py` — test-only interventions: `Identity` (records its input), `ZeroOut`, `Replace` (applies a function once the sequence reaches a length).
- `test_model_config_loading.py` — YAML loading, `prefit_lens` parsing, `--device` override, profile-name check, sizes read from the model, left padding, `token_ids`, and the post-norm `hidden_states` fact.
- `test_model_hooks.py` — identity vs zero-out, hook removal on exit and exception, capture shapes, `unembed(last capture) == logits`, post-intervention capture, gradients through interventions, tensor/tuple block-output normalisation, `PerRow` edits only its own row, passes gradients, and rejects a wrong batch size.
- `test_model_inference.py` — chat template and kwargs, left padding, batched equals single (logits, loss, generation), `generate` returns only new text and stops at EOS, `greedy_tokens` calls its provider once per step with the grown sequence, `generate` equals the pre-refactor loop, `sequence_nll` sums and `encode(max_length)` truncation.
- `test_model_lora_training.py` — collate text and labels (generic `to_text`), batch cycling, hooks still fire after LoRA, fp32 adapter over bf16 base, `adapter_disabled`, save/load round trip, wrong-profile load raises, only adapters change, loss falls, factory called once per batch, interventions sit in the training graph.
- `test_jspace_model.py` — `ProjectOut` on a left-padded batch removes the vectors at every editable position and leaves each row's first real token and its padding unchanged; on one unpadded row the edit is bit-identical to editing `h[:, skip_first:]`; `lens_view` restores `requires_grad` and train mode before and after `attach_lora`; jlens's encoding equals `lm.token_ids` for Qwen; `lens_logits` applies interventions inside jlens's forward; `load_lens` rejects a wrong `d_model`, missing layers and a profile with no lens, and accepts a fitting lens.
- `test_jspace_eval.py` — model-free: loaders, split leakage, answer matching, probe separability and subject leakage, decomposition recovery, projection, and swap math.
- `test_evaluation_model.py` — `greedy_continuation` returns the decoded tokens and the first id; every builder returns a sequence (`()` for `none`) on band layers; random builders repeat under the same `item_rng`; perplexity is token-weighted with truncation and short texts skipped.
- `test_finetune.py` — `item_text`, training conditions are eval conditions, and training under `condition_factory("bridge")` with a synthetic lens calls the factory once per batch with one `PerRow` of per-row `ProjectOut`s, the loss falls, and only adapter weights change.
