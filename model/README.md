# jbypass/model

Loads a causal LM described by a model profile and provides inference, LoRA fine-tuning,
and a hook framework for reading and editing the residual stream. The J-space maths
(J-lens, ablation directions) lives elsewhere and plugs in through `hooks.Intervention`.

Nothing here names a model attribute except `loading.py`, which reads the paths from the
profile. Switching models is `model:` in `configs/run.yaml` or `--model` on the scripts.

## Files and functions

### `config.py` — settings dataclasses and YAML loaders
- `ModelProfile`: every model-specific value (`name`, `hf_id`, `dtype`, `device`, `blocks_path`, `final_norm_path`, `chat_template_kwargs`, `lora_target_modules`). One YAML file per model in `configs/models/`.
- `GenerationSettings` (`max_new_tokens`), `LoraSettings` (`r`, `alpha`, `dropout`), `TrainSettings` (`learning_rate`, `batch_size`, `steps`, `seed`).
- `RunConfig`: chosen profile name, `profiles_dir`, and the three settings above.
- `load_run_config(path)`: reads run.yaml; `profiles_dir` is resolved relative to the file.
- `load_model_profile(name, profiles_dir)`: reads `<profiles_dir>/<name>.yaml`; raises if the file's `name` differs.

### `loading.py` — the model wrapper
- `LanguageModel(hf_model, tokenizer, profile)`: wraps an already-built HF model (tests use this with tiny random models). Sets left padding on the tokenizer.
  - `hf_model`: what forward passes call; becomes a `PeftModel` after `attach_lora`/`load_adapter`.
  - `blocks`: decoder blocks (hook targets), resolved on the original model, so still valid after LoRA.
  - `n_layers`, `d_model`, `device`, `dtype`: read from the loaded model, never from config.
  - `unembed(resid)`: `lm_head(final_norm(resid))`.
  - `unembedding_matrix`: `[vocab, d_model]` lm_head weight.
  - `token_ids(text)`: encoding without added special tokens.
- `load_model(profile)`: loads tokenizer and weights from the HF cache (downloading if needed) onto the profile's device and dtype, in eval mode.

### `hooks.py` — the intervention contract
- Hook point "layer i" = the residual returned by `lm.blocks[i]`.
- `HookContext(layer, attention_mask)`: passed to every intervention call.
- `Intervention` (Protocol): has `layers`; `__call__(resid, ctx) -> resid` on `[batch, seq, d_model]`.
- `apply_interventions(lm, interventions, attention_mask)`: context manager; registers forward hooks, removes them on exit or exception. Raises if an intervention changes shape or dtype.
- `capture_residuals(lm, layers)`: context manager yielding `{layer: resid}` (in-graph; the latest forward pass wins).
- `block_residual(output)` / `with_block_residual(output, resid)`: unwrap/rewrap a block output that is a tensor or a tuple with the residual first.

### `prompts.py` — input formatting
- `chat_prompt(lm, question)`: one user turn through the tokenizer's chat template, with the assistant prefix and the profile's `chat_template_kwargs` (Qwen3: `enable_thinking=False`). Formatters have the signature `(lm, question) -> str`.

### `inference.py` — forward-only use (all public functions run under `no_grad`)
- `Encoded(input_ids, attention_mask)`; `encode(lm, texts)`: left-padded, on device, no added special tokens.
- `position_ids(attention_mask)`: positions counted from each row's first real token, so padding does not shift them.
- `forward_logits(lm, input_ids, attention_mask, interventions=())`: in-graph logits with interventions applied (shared with training).
- `nll_per_row(logits, labels)`: summed fp32 next-token NLL and scored-token count per row; labels of `IGNORE_INDEX` (-100) are skipped.
- `generate(lm, prompts, settings, interventions=())`: greedy; returns only new text, stopped at EOS. No KV cache (each step reruns the full sequence, so hooks always see the whole sequence); recorded as `TODO(debt)`.
- `next_token_logits(lm, prompts, interventions=())`: `[batch, vocab]`.
- `sequence_loss(lm, texts, interventions=())`: `[batch]` mean NLL over each text's tokens after the first (exp gives perplexity).

### `data.py` — training examples
- `Example(question, answer, bridge=None)`: the shape shared with the dataset/eval code.
- `Batch(input_ids, attention_mask, labels, examples)`: left-padded; labels mask prompt and padding.
- `example_ids(lm, example, format_prompt)`: ids = `token_ids(prompt) + token_ids(answer + eos)`, so the prompt is tokenised exactly as at inference; only answer and EOS are scored.
- `collate(lm, examples, format_prompt)`: builds a `Batch` on the model's device.

### `lora.py` — adapter lifecycle
- `attach_lora(lm, settings)`: fresh trainable adapter on the profile's `lora_target_modules`; base frozen; adapter weights fp32 over a bf16 base.
- `save_adapter(lm, path)`: adapter weights plus `jbypass_profile.json` with the profile name.
- `load_adapter(lm, path)`: loads a frozen adapter (for inference); raises `ValueError` if the profile name differs.
- `adapter_disabled(lm)`: context manager that runs the exact base model.

### `training.py` — the optimisation loop
- `InterventionFactory = Callable[[Batch], Sequence[Intervention]]`; `StepCallback = Callable[[step, loss, lm], None]`.
- `no_interventions(batch)` / `ignore_step(...)`: the defaults (no-ablation control; no callback).
- `example_batches(examples, settings)`: `settings.steps` batches from a seeded reshuffle each pass.
- `batch_loss(lm, batch, interventions)`: in-graph mean NLL over answer tokens, fp32.
- `train(lm, examples, settings, format_prompt, intervention_factory=no_interventions, on_step=ignore_step) -> list[float]`: AdamW over every parameter with `requires_grad` (LoRA or full fine-tune alike); calls the factory once per batch; leaves the model in eval mode with no `.grad` values.

## Contract rules (for the J-space and eval modules)
- Hook points are defined by these hooks, not by `output_hidden_states`. In transformers 5.x the last `hidden_states` entry is post-norm, so the two disagree at the final layer.
- Row `i` of `resid` belongs to `batch.examples[i]`; that is how a factory builds per-example ablations.
- An intervention returns a new tensor in `resid.dtype` and never edits in place. Do projection maths in fp32 (the base model is bf16).
- A capture opened inside `apply_interventions` records the post-intervention value.
- Positions where `ctx.attention_mask == 0` are left padding; exclude them from averages.
- An `on_step` callback must leave the model in `train()` mode and must not leave values in parameter `.grad`.
- Layer choices should be expressed against `lm.n_layers`, never a fixed number.
- Saved artefacts tied to a model (J-lens files) should record `lm.profile.name`, as adapters do.
