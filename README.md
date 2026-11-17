# JBypass

J-Bypassing: can a model learn while under J-lens ablations? See the proposal PDF.

## Setup (Windows, miniconda)

```powershell
C:\Users\archi\miniconda3\Scripts\conda.exe env create -f environment.yml
$py = "C:\Users\archi\miniconda3\envs\jbypass\python.exe"
& $py -m pip install -e .
& $py -m model.run_download                 # model named in configs/run.yaml
& $py -m pytest tests                       # CPU, tiny random models
& $py -m model.run_smoke                    # GPU end-to-end check; add --model qwen3-0.6b to switch
```
For the viewer's translations, save `cedict_1_0_ts_utf-8_mdbg.txt.gz` from https://www.mdbg.net/chinese/dictionary?page=cc-cedict into `downloads/` by hand in a browser (scripted download is prohibited by the site; see `commands.md`).

## Experiment
```powershell
& $py -m evaluation.run_lens_checks --tag M                                   # lens behaves like the paper?
& $py -m evaluation.make_splits --out results/splits.json                    # items M solves, split by bridge
& $py -m finetune.run_train --condition bridge --split_file results/splits.json --out runs/mexp    # Mexp
& $py -m finetune.run_train --condition none   --split_file results/splits.json --out runs/mctrl   # Mctrl
& $py -m evaluation.run_eval --adapter runs/mexp --split_file results/splits.json --split test_seen --tag mexp
```
Every command takes `--config`, `--model` and `--device` (default: the profile's device). See `evaluation/README.md` for the full eval run order.

## Layout
Dependencies run one way: `model` ← `jspace` ← `evaluation` ← `finetune` / `viz`.
- `model/` — loading, residual-stream hooks, inference, LoRA, training loop, CLI helpers, `run_download` and `run_smoke`.
- `jspace/` — attaches a J-lens to a loaded model: lens loading, J-lens vectors, interventions, readout, `fit_lens`.
- `evaluation/` — dataset, ablation conditions, scoring, runner, lens checks, mechanism, probes, perplexity.
- `finetune/` — LoRA training under an eval condition; saves the adapter the eval scripts load.
- `viz/` — local Gradio app: run one example under one ablation and browse the J-lens readout of every layer (`python -m viz.run_app`).
- `configs/` — `run.yaml` (which model plus run settings), `viz.yaml` (viewer settings) and `models/<name>.yaml` profiles.
- `tests/` — flat unit tests with one shared `conftest.py`.
- `commands.md` — every CLI command (setup, checks, splits, training, eval, lens refit) in run order.
- `debt.md` — recorded technical debt that spans several files.
- `environment.yml`, `pyproject.toml` — the `jbypass` env (with the pinned `jlens` commit) and package definition.
