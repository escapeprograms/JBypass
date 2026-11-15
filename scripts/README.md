# scripts

Command-line entry points. Each takes `--config` (default `configs/run.yaml`) and `--model <profile>` (overrides run.yaml).

- `_cli.py` — shared argument parsing: `parse_run_args(description)` and `resolve_run(args) -> (RunConfig, ModelProfile)`.
- `download_model.py` — `main()`: downloads the profile's model into the default Hugging Face cache.
- `smoke_test.py` — end-to-end GPU check on a real model. `check_generation` (answer, then a zero-out rerun that must differ), `check_capture` (residuals at every layer have shape `[1, seq, d_model]`), `check_lora` (a few LoRA steps with a dummy identity-intervention factory; `adapter_disabled` must reproduce the pre-training logits). Prints losses and peak VRAM; raises on the first failed check. `ZeroOut` and `Identity` are dummy interventions used only here.
