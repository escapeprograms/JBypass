# model

Loads a causal LM described by a model profile and provides inference, LoRA fine-tuning,
and a hook framework for reading and editing the residual stream. It knows nothing about
lenses or datasets: the J-space maths lives in `jspace/` and plugs in through
`hooks.Intervention`. Dependencies run one way: `model` ← `jspace` ← `evaluation` ← `finetune`.

Nothing here names a model attribute except `loading.py`, which reads the paths from the
profile. Switching models is `model:` in `configs/run.yaml` or `--model` on any command.

## Commands
```
python -m model.run_download [--model qwen3-0.6b]   # fetch weights into the HF cache
python -m model.run_smoke    [--model qwen3-0.6b]   # GPU end-to-end check
```

## Files and functions

### `config.py` — settings dataclasses and YAML loaders
- `LensSource(repo, file, revision)`: Hub location of a prefit J-lens.
- `ModelProfile`: every model-specific value (`name`, `hf_id`, `dtype`, `device`, `blocks_path`, `final_norm_path`, `chat_template_kwargs`, `lora_target_modules`, `prefit_lens`). One YAML file per model in `configs/models/`; `prefit_lens: null` when the model has none.
- `GenerationSettings` (`max_new_tokens`), `LoraSettings` (`r`, `alpha`, `dropout`), `TrainSettings` (`learning_rate`, `batch_size`, `steps`, `seed`).
- `RunConfig`: chosen profile name, `profiles_dir`, the three settings above, `jspace` (the raw optional `jspace:` section as a dict, `{}` when absent; model/ does not interpret it, `jspace.jspace_config` does) and `config_dir` (the YAML file's resolved folder, base for relative paths in it).
- `load_run_config(path)`: reads run.yaml; `profiles_dir` is resolved relative to the file. Marked `TODO(debt-002)`: eval and fine-tune settings are not sections of it yet.
- `load_model_profile(name, profiles_dir)`: reads `<profiles_dir>/<name>.yaml` (turning `prefit_lens` into a `LensSource`); raises if the file's `name` differs.

### `loading.py` — the model wrapper
- `LanguageModel(hf_model, tokenizer, profile)`: wraps an already-built HF model (tests use this with tiny random models). Sets left padding on the tokenizer.
  - `hf_model`: what forward passes call; becomes a `PeftModel` after `attach_lora`/`load_adapter`.
  - `base_model`: the original HF model (PEFT edits its modules in place); what `jlens.from_hf` is given.
  - `blocks`: decoder blocks (hook targets), resolved on the original model, so still valid after LoRA.
  - `n_layers`, `d_model`, `device`, `dtype`: read from the loaded model, never from config.
  - `unembed(resid)`: `lm_head(final_norm(resid))`. `unembedding_matrix`: `[vocab, d_model]` lm_head weight.
  - `token_ids(text)`: encoding without added special tokens.
- `freeze(lm)`: turns off `requires_grad` on every parameter (eval and lens fitting; the eval code relies on frozen weights so maths on the unembedding builds no autograd graph).
- `load_tokenizer(profile)`: the profile's tokenizer only (HF cache, downloading if needed); no weights. Used by the viewer to list questions before any model is loaded.
- `load_model(profile)`: `load_tokenizer` plus the weights from the HF cache (downloading if needed) onto the profile's device and dtype, in eval mode.

### `hooks.py` — the intervention contract
- Hook point "layer i" = the residual returned by `lm.blocks[i]`.
- `HookContext(layer, attention_mask)`: passed to every intervention call.
- `Intervention` (Protocol): has `layers`; `__call__(resid, ctx) -> resid` on `[batch, seq, d_model]`.
- `PerRow(rows)`: an `Intervention` where `rows[i]` applies to batch row `i` only (each row's interventions see that row's slice of the residual and mask). Raises if the batch size differs from `len(rows)`.
- `apply_interventions(lm, interventions, attention_mask)`: context manager; registers forward hooks, removes them on exit or exception. Raises if an intervention changes shape or dtype.
- `capture_residuals(lm, layers)`: context manager yielding `{layer: resid}` (in-graph; the latest forward pass wins).
- `block_residual(output)` / `with_block_residual(output, resid)`: unwrap/rewrap a block output that is a tensor or a tuple with the residual first.

### `prompts.py` — chat formatting
- `chat_prompt(lm, question)`: one user turn through the tokenizer's chat template, with the assistant prefix and the profile's `chat_template_kwargs` (Qwen3: `enable_thinking=False`). Used by the smoke test; the J-space experiment uses raw `Fact: ...` prompts.

### `inference.py` — forward passes and decoding (public functions run under `no_grad`, except the two in-graph helpers)
- `Encoded(input_ids, attention_mask)`; `encode(lm, texts, max_length=None)`: left-padded, on device, no added special tokens; truncates each text to `max_length` tokens if given.
- `InterventionsFor = Callable[[Encoded], Sequence[Intervention]]`: builds interventions from the current sequence.
- `position_ids(attention_mask)`: positions counted from each row's first real token, so padding does not shift them.
- `forward_logits(lm, input_ids, attention_mask, interventions=())`: in-graph logits with interventions applied (shared with training).
- `nll_per_row(logits, labels)`: in-graph summed fp32 next-token NLL and scored-token count per row; labels of `IGNORE_INDEX` (-100) are skipped.
- `next_token_logits(lm, prompts, interventions=())`: `[batch, vocab]`.
- `forward_residuals(lm, enc, layers, interventions=())`: `{layer: [batch, seq, d]}` residuals of one forward pass, post-intervention (the capture is opened inside `apply_interventions`), in the model's dtype.
- `sequence_nll(lm, enc, interventions=())`: summed NLL and token count per row over each row's real tokens after the first.
- `sequence_loss(lm, texts, interventions=())`: `[batch]` mean NLL (exp gives perplexity).
- `greedy_tokens(lm, enc, max_new_tokens, interventions_for)`: `[batch, n_new]` greedy tokens; calls `interventions_for` before every step with the grown sequence; stops once every row has produced EOS. No KV cache (`TODO(debt)`).
- `generate(lm, prompts, settings, interventions=())`: greedy text continuation built on `greedy_tokens`; new text only, stopped at EOS.

### `data.py` — training batches, generic over the example type
- `ExampleText = Callable[[LanguageModel, T], tuple[str, str]]`: turns an example into `(prompt, completion)`.
- `Batch(input_ids, attention_mask, labels, examples)`: left-padded; labels mask prompt and padding; `examples` are passed through untouched.
- `example_ids(lm, example, to_text)`: ids = `token_ids(prompt) + token_ids(completion + eos)`, so the prompt is tokenised exactly as at inference; only completion and EOS are scored.
- `collate(lm, examples, to_text)`: builds a `Batch` on the model's device.

### `lora.py` — adapter lifecycle
- `attach_lora(lm, settings)`: fresh trainable adapter on the profile's `lora_target_modules`; base frozen; adapter weights fp32 over a bf16 base.
- `save_adapter(lm, path)`: adapter weights plus `jbypass_profile.json` with the profile name.
- `saved_profile(path)`: the profile name recorded in a saved adapter's `jbypass_profile.json`.
- `load_adapter(lm, path)`: loads a frozen, unmerged adapter (every eval script uses this); raises `ValueError` if the profile name differs.
- `adapter_disabled(lm)`: context manager that runs the exact base model.

### `training.py` — the optimisation loop
- `InterventionFactory = Callable[[Batch], Sequence[Intervention]]`; `StepCallback = Callable[[step, loss, lm], None]`.
- `no_interventions(batch)` / `ignore_step(...)`: the defaults (no-ablation control; no callback).
- `example_batches(examples, settings)`: `settings.steps` batches from a seeded reshuffle each pass.
- `batch_loss(lm, batch, interventions)`: in-graph mean NLL over completion tokens, fp32.
- `train(lm, examples, settings, to_text, intervention_factory=no_interventions, on_step=ignore_step) -> list[float]`: AdamW over every parameter with `requires_grad`; calls the factory once per batch; leaves the model in eval mode with no `.grad` values.

### `cli.py` — flags shared by every command
- `add_profile_args(parser)`: `--config` (default `configs/run.yaml`), `--model` (overrides run.yaml), `--device` (overrides the profile's device; default = the profile's).
- `resolve_profile(args) -> (RunConfig, ModelProfile)`: loads run.yaml and the chosen profile with the device override applied.

### `run_download.py`, `run_smoke.py` — commands
- `run_download.main()`: downloads the profile's model into the default Hugging Face cache.
- `run_smoke.main()`: end-to-end GPU check. `check_generation` (answer, then a zero-out rerun that must differ), `check_capture` (residuals at every layer have shape `[1, seq, d_model]`), `check_lora` (a few LoRA steps on chat-formatted toy questions via `toy_text`, with a dummy identity-intervention factory; `adapter_disabled` must reproduce the pre-training logits). Prints losses and peak VRAM; raises on the first failed check. `ZeroOut` and `Identity` are dummy interventions used only here.

## Contract rules (for the J-space and eval modules)
- Hook points are defined by these hooks, not by `output_hidden_states`. In transformers 5.x the last `hidden_states` entry is post-norm, so the two disagree at the final layer.
- Row `i` of `resid` belongs to `batch.examples[i]`; `PerRow` is how a factory applies per-example ablations.
- An intervention returns a new tensor in `resid.dtype` and never edits in place. Do projection maths in fp32 (the base model is bf16).
- A capture opened inside `apply_interventions` records the post-intervention value.
- Positions where `ctx.attention_mask == 0` are left padding; never edit them and exclude them from averages.
- An `on_step` callback must leave the model in `train()` mode and must not leave values in parameter `.grad`.
- Layer choices should be expressed against `lm.n_layers`, never a fixed number.
- Saved artefacts tied to a model (J-lens files) should record `lm.profile.name`, as adapters do.
