# configs

- `run.yaml` — the run: which model profile (`model`), where profiles live (`profiles_dir`, relative to this folder), and generation, LoRA and training settings (`finetune.run_train` reads `lora` and `train`). Memory-bound values (batch size) live here. Pass another file with `--config` to any command.
- `models/<name>.yaml` — one model profile per file; the only place model-specific values live (HF id, dtype, device, block and final-norm paths, chat-template kwargs, LoRA target modules, and `prefit_lens`: the Hub `repo`, `file` and `revision` of the model's prefit J-lens, or `null`). The file name must equal its `name`. `--device` on a command overrides `device`.
  - `qwen3-1.7b.yaml` — default model; prefit lens from `neuronpedia/jacobian-lens`.
  - `qwen3-0.6b.yaml` — second profile, proves switching is config-only. No prefit lens exists (`prefit_lens: null`): J-space commands need `--lens_path` to a lens from `jspace.fit_lens`.

- `viz.yaml` — the J-space viewer (`viz/`); paths relative to this folder.
  - `runs_dir` — folder whose subfolders holding a saved adapter become model choices.
  - `lenses_dir` — folder of refit lenses (`../lenses`): `<adapter folder name>.pt` for an adapter, `<profile>.pt` for a base model; read at startup, a model's file (if present) is its second, readout-only lens.
  - `split_file` — `{split: [item names]}` JSON for the split dropdown, or `null` (only `all`); a missing file raises.
  - `dictionary_file` — the gzipped CC-CEDICT for the click-to-translate box (`../downloads/cedict_1_0_ts_utf-8_mdbg.txt.gz`, saved by hand from mdbg.net; see `viz/README.md`), or `null` (no translations). Configured but missing raises at startup.
  - `top_n` — top tokens kept per (position, layer) cell.
  - `max_tracked` — cap on frequently-top tokens given rank charts, besides the pinned bridge/answers; `null` = all (pages can reach ~10 MB).
  - `page_height` — heatmap iframe height in pixels.
  - `server_name`, `server_port` — where the app listens.

- `run.yaml` may hold an optional `jspace:` section: overrides of any `JSpaceConfig` field (`jspace/config.py`), applied by `jspace.jspace_config` for every command (training, readout stats, evals, viewer). Unknown keys raise. `band_layers: [first, last]` (inclusive layer indices) replaces the default band fractions (layers 11-26 on qwen3-1.7b); `readout_stats_dir` is relative to this folder (default without it: `stats/` under the working directory); `fit_checkpoint_every` (prompts between lens-fit checkpoint writes, default 10) takes an integer or `null` (write only at the end). `run.yaml` has no section, so the defaults (exp 2) apply.
- `exp3.yaml` — experiment 3: `run.yaml`'s values plus `jspace: {band_layers: [8, 26], readout_stats_dir: ../stats/exp3}` (clamp band from layer 8; its statistics go to `stats/exp3/<model>.pt`). Pass `--config configs/exp3.yaml` to every command of the experiment.

Eval and fine-tune settings are still dataclass defaults (`evaluation/config.py`, `finetune/config.py`); see debt 002.
