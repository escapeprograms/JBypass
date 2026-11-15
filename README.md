# JBypass

J-Bypassing: can a model learn while under J-lens ablations? See the proposal PDF.

## Setup (Windows, miniconda)

```powershell
C:\Users\archi\miniconda3\Scripts\conda.exe env create -f environment.yml
$py = "C:\Users\archi\miniconda3\envs\jbypass\python.exe"
& $py -m pip install -e .
& $py scripts\download_model.py            # model named in configs/run.yaml
& $py -m pytest tests                       # CPU, tiny random models
& $py scripts\smoke_test.py                 # GPU end-to-end check; add --model qwen3-0.6b to switch
```

## Layout
- `jbypass/` — the Python package (`jbypass/model/`: loading, hooks, inference, LoRA, training).
- `configs/` — `run.yaml` (which model plus run settings) and `models/<name>.yaml` profiles.
- `scripts/` — command-line entry points.
- `tests/` — unit tests.
- `debt.md` — recorded technical debt that spans several files.
- `environment.yml`, `pyproject.toml` — the `jbypass` env and package definition.
