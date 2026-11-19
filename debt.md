# Technical debt

Debt that spans several files. Single-spot debt stays as `# TODO(debt): ...` in the code.
Mark each affected spot with `# TODO(debt-NNN)` using the entry's number.
Before starting a task, check whether any entry's "Fix when" matches it.

## Template

```
## NNN: <short title>
- Where: <files / functions marked TODO(debt-NNN)>
- What: <the shortcut taken and why>
- Cost: <what it makes harder or wrong>
- Fix: <the intended fix>
- Fix when: <the task or trigger that should pay it off>
- Added: <YYYY-MM-DD>
```

## Entries

## 001: jlens tokenises prompts on its own
- Where: `jspace/loading.py` (`lens_view`), `jspace/readout.py` (`_unpadded_mask`), `jspace/fit_lens.py` (`main`), `evaluation/mechanism.py` (`question_start`), all marked `TODO(debt-001)`.
- What: `lens.apply` and `jlens.fit` encode prompts with jlens's own `HFLensModel.encode` (special tokens on, and `from_hf`'s `force_bos` may set `tokenizer.add_bos_token`), not with `model.inference.encode` (no special tokens). We mirror its encoding to build the all-ones mask for interventions and to index positions. For Qwen the two are equal (no BOS token; `tests/test_jspace_model.py` checks it).
- Cost: for a tokenizer that adds a BOS token, jlens's sequences gain one leading token: positions from `question_start` and builders that use our `Encoded` would no longer line up with jlens's readout, and `skip_first` would count the BOS.
- Fix: run lens readouts on our own `Encoded` (call the residual capture and `lens.transport` / `unembed` ourselves) or pass `force_bos=False` and add the BOS in `model.inference.encode` consistently. The building blocks for the first option now exist: `model.inference.forward_residuals` + `jspace.readout.residual_logits`, used by `viz/slice.py`.
- Fix when: a profile is added for a model whose tokenizer adds a BOS token.
- Added: 2026-11-15

## 002: two config styles
- Where: `model/config.py` (`load_run_config`), `evaluation/config.py` (`EvalConfig`), `finetune/config.py` (`FinetuneConfig`), `viz/run_app.py` (`build_context`), all marked `TODO(debt-002)`.
- What: run, model and J-space settings are YAML (`configs/`; J-space overrides in run.yaml's optional `jspace:` section, read by `jspace.jspace_config`); eval and fine-tune settings (`EvalConfig`, `FinetuneConfig`) are still dataclass defaults built in code.
- Cost: a run cannot vary eval settings (conditions, seeds, readout_k, datasets, out_dir default) or fine-tune settings (dataset, split) without editing code, and those settings are not recorded in the run's file.
- Fix: add `eval:` / `finetune:` sections to run.yaml, kept raw in `RunConfig` like `jspace`, and build the configs with `eval_config(run)` / `finetune_config(run)` in `jspace_config`'s style; pass them down instead of `EvalConfig()` / `FinetuneConfig()`.
- Fix when: a run needs to vary eval or fine-tune settings without editing code.
- Added: 2026-11-15 (J-space half paid off 2026-11-16)

## 003: lens files carry no provenance
- Where: `jspace/fit_lens.py` (`fit_to_file`), `viz/catalog.py` (`refit_lens_file`), `viz/session.py` (`attach_refit`), all marked `TODO(debt-003)`.
- What: a lens and its model are paired by file name only (`lenses/<adapter folder name>.pt` or `lenses/<profile>.pt`; base profiles and adapter folders share one namespace), and jlens's checkpoint resume (`<out>.ckpt`) does not check the model either.
- Cost: a retrained adapter is read through, or resumes the fit of, a stale lens without any error. The original lens and the refits also differ in recipe (Neuronpedia: 466 prompts with early stopping; ours: a fixed prompt count), which nothing records.
- Fix: a sidecar JSON written by `fit_to_file` (profile, adapter, recipe), checked on load and on resume.
- Fix when: an adapter is retrained into an existing folder, or lens comparisons go into a write-up.
- Added: 2026-11-19
