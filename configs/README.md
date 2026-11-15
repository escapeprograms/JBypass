# configs

- `run.yaml` — the run: which model profile (`model`), where profiles live (`profiles_dir`, relative to this folder), and generation, LoRA and training settings (`finetune.run_train` reads `lora` and `train`). Memory-bound values (batch size) live here. Pass another file with `--config` to any command.
- `models/<name>.yaml` — one model profile per file; the only place model-specific values live (HF id, dtype, device, block and final-norm paths, chat-template kwargs, LoRA target modules, and `prefit_lens`: the Hub `repo`, `file` and `revision` of the model's prefit J-lens, or `null`). The file name must equal its `name`. `--device` on a command overrides `device`.
  - `qwen3-1.7b.yaml` — default model; prefit lens from `neuronpedia/jacobian-lens`.
  - `qwen3-0.6b.yaml` — second profile, proves switching is config-only. No prefit lens exists (`prefit_lens: null`): J-space commands need `--lens_path` to a lens from `jspace.fit_lens`.

J-space and eval settings are still dataclass defaults (`jspace/config.py`, `evaluation/config.py`, `finetune/config.py`); see debt 002.
