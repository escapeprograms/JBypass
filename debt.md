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
- Where: `model/config.py` (`load_run_config`), `jspace/config.py` (`JSpaceConfig`), `evaluation/config.py` (`EvalConfig`), `evaluation/cli_common.py` (`load_frozen_jspace`), `finetune/run_train.py` (`main`), `viz/run_app.py` (`build_context`), all marked `TODO(debt-002)`.
- What: run and model settings are YAML (`configs/`); J-space and eval settings are dataclass defaults built in code.
- Cost: a run cannot vary band, top-k, seeds or eval settings without editing code, and a run's settings are not recorded in one file.
- Fix: load `JSpaceConfig` and `EvalConfig` from sections of run.yaml (or a separate YAML) in `model.cli.resolve_profile`'s style and pass them down.
- Fix when: a run needs to vary J-space or eval settings without editing code.
- Added: 2026-11-15
