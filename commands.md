# Commands

Every command runs from the repo root in the `jbypass` env. On Windows, without activating the env:
```powershell
$py = "C:\Users\archi\miniconda3\envs\jbypass\python.exe"   # then use `& $py -m ...` in place of `python -m ...`
```
With the env active (`conda activate jbypass`), plain `python -m ...` works. Add `--help` to any command to list its flags.

**Model names used below:** M is the base model with no fine-tuning. Mexp is fine-tuned while the bridge is ablated (the experiment). Mctrl is fine-tuned with no ablation (control). Mrand is fine-tuned under a random ablation of the same size (control).

## Shared flags

**Every command** takes these:
| flag | default | what it does |
|---|---|---|
| `--config <path>` | `configs/run.yaml` | Run settings file: which model, generation length, LoRA rank, training steps, batch size, learning rate, seed, and an optional `jspace:` section that overrides J-space settings (any field of `JSpaceConfig` in `jspace/config.py`), e.g. `band_layers: [8, 26]` (inclusive layer indices, replacing the default band = layers 11-26), `clamp_z`, `readout_stats_dir` (relative to the YAML's folder). Unknown keys stop with an error. Training, readout statistics, every eval command and the viewer all read it, so one `--config` fixes the band for a whole experiment. `configs/run.yaml` has no `jspace:` section (defaults = exp 2); `configs/exp3.yaml` is experiment 3 (section 9). |
| `--model <name>` | `model:` in run.yaml (`qwen3-1.7b`) | Model profile from `configs/models/<name>.yaml`. Overrides run.yaml. Options: `qwen3-1.7b`, `qwen3-0.6b`. |
| `--device <name>` | the profile's `device` (`cuda`) | Where to run: `cuda`, `cpu` or `mps` (Mac). |

**Every eval command** (`evaluation.*`) also takes these:
| flag | default | what it does |
|---|---|---|
| `--tag <name>` | `M` | A label for the run. It is printed in the log and used in the output filename (`<out_dir>/eval_<tag>_...json`), so runs don't overwrite each other. Use `M` for the base model and the adapter's name for a fine-tuned one (`mexp`, `mctrl`, ...). It does not change what gets computed. |
| `--adapter <dir>` | none (base model M) | Loads a LoRA adapter saved by `finetune.run_train`, so you evaluate Mexp, Mctrl or Mrand instead of M. |
| `--lens_path <file.pt>` | the profile's prefit lens (downloaded from Neuronpedia) | Uses a different J-lens for the ablations and readouts, e.g. one refit with `jspace.fit_lens`. |
| `--stats_path <file.pt>` | `<readout_stats_dir>/<model>.pt` if it exists, else none (`stats/qwen3-1.7b.pt` with run.yaml, `stats/exp3/qwen3-1.7b.pt` with exp3.yaml) | Reference readout statistics from `evaluation.run_readout_stats` (section 3b). Needed only by the clamp conditions; an adapter uses base M's file (stats are keyed by model profile). |
| `--limit <N>` | all items | Uses only the first N items of each dataset. For quick test runs. |
| `--out_dir <dir>` | `results` | Folder the result JSON is written to. Give each run (experiment) its own folder, e.g. `--out_dir "results/exp 3"`; it is created if missing. Existing layout: `results/baseline/` (base M), `results/exp 1/` (Mexp), `results/exp 2/` (Mclamp and the clamp_z sweep on M), `results/smoke/`. |

**Splits flags** (`run_eval`, `run_mechanism`):
| flag | default | what it does |
|---|---|---|
| `--split_file <path>` | none (whole dataset) | JSON file written by `evaluation.make_splits`. |
| `--split <name>` | none | Which part of the split file to use: `train`, `test_seen` (new questions about bridges seen in training) or `test_unseen` (bridges never seen in training). Needs `--split_file`. |

---

## 1. Setup (once)
```powershell
conda env create -f environment.yml
conda activate jbypass
pip install -e .
python -m model.run_download                    # downloads the model named in configs/run.yaml
python -m model.run_download --model qwen3-0.6b # downloads another profile
```
`model.run_download` saves the weights and tokenizer to the Hugging Face cache. It takes only the shared flags.

For the viewer's click-to-translate box (section 8), download the CC-CEDICT dictionary **by hand in a browser**: open https://www.mdbg.net/chinese/dictionary?page=cc-cedict, download `cedict_1_0_ts_utf-8_mdbg.txt.gz`, and save it as `downloads/cedict_1_0_ts_utf-8_mdbg.txt.gz` in the repo root (gitignored). There is no download command: the site states that automated or scripted access is prohibited. Without the file, `viz.run_app` stops at startup; set `dictionary_file: null` in `configs/viz.yaml` to run without translations.

## 2. Checks
```powershell
python -m pytest tests                          # unit tests on CPU with tiny random models; no download needed
python -m model.run_smoke                       # end-to-end GPU check: generation, hooks, residual capture, a few LoRA steps
python -m evaluation.run_lens_checks --tag M    # does the lens reproduce the paper's results on this model?
```
- `run_smoke` takes only the shared flags. It stops with an error at the first check that fails.
- `run_lens_checks` reports the bridge's lens readout (pass@k), clean and with the bridge ablated, plus the paper's swap check. It writes `<out_dir>/lens_checks_<tag>.json` (default `results/`).

## 3. Data and splits
```powershell
python -m evaluation.data.build_jbypass_dataset               # rebuilds evaluation/data/jbypass_two_hop.json (already in the repo)
python -m evaluation.make_splits --out results/splits.json    # train / test splits
```
`make_splits` keeps the jbypass items that M answers correctly on all three prompt types, then splits them by bridge.
| flag | default | what it does |
|---|---|---|
| `--out <path>` | `results/splits.json` | Where to write the split file. |
| `--unseen_frac <float>` | `0.3` | Share of bridges held out entirely for `test_unseen`. The remaining bridges' items are split into `train` and `test_seen`. |

It also takes the eval flags (`--adapter`, `--lens_path`, `--limit`). Normally you run it on base M with no extra flags.

## 3b. Readout statistics (for the clamp conditions)
```powershell
python -m evaluation.run_readout_stats                                            # -> stats/qwen3-1.7b.pt
python -m evaluation.run_readout_stats --config configs/exp3.yaml                 # band 8-26 -> stats/exp3/qwen3-1.7b.pt
python -m evaluation.run_readout_stats --split_file results/splits.json --split train --out stats/train.pt
```
The clamp exists because the Mexp adapter answered better with the bridge ablation on than off (0.92 vs 0.66 on test_seen), so it was keying on the exactly-zero bridge readout that `bridge` produces and a clean pass never does; `bridge_clamp` instead lowers the readout only to its normal level. `run_readout_stats` records that normal level: for every token and every band layer, the mean and standard deviation of its J-lens logit over the dataset's prompts (each prompt leaves out its own bridge's tokens). The clamp caps the bridge at mean + `clamp_z` x std (`JSpaceConfig.clamp_z`, 0.5; override with `jspace: {clamp_z: ...}` in the `--config` file). Statistics cover only the band's layers, so run it once per model profile and band (i.e. per `--config` whose band differs), before evaluating or training with `bridge_clamp` / `wrong_bridge_clamp`; every later command with the same `--config` loads `<readout_stats_dir>/<model>.pt` automatically. It prints the number of texts, editable positions, band layers and the output path.
| flag | default | what it does |
|---|---|---|
| `--dataset <name>` | `jbypass` | Dataset whose prompts are the reference texts. |
| `--split_file <path>`, `--split <name>` | none (whole dataset) | Restrict the reference texts to one split. |
| `--out <path>` | `<readout_stats_dir>/<model>.pt` (`stats/` by default, `stats/exp3/` with exp3.yaml) | Where to write the statistics. Another path must then be passed as `--stats_path` to later commands. |

It also takes the shared and eval flags (`--adapter`, `--lens_path`, `--limit`; `--stats_path` is accepted but not needed). The command also loads the existing default file of its `--config`, so if `<readout_stats_dir>/<model>.pt` no longer fits that config's band, delete it before rerunning. A config with a different band should set its own `readout_stats_dir` (as `configs/exp3.yaml` does with `stats/exp3/`) so each band keeps its own file.

## 4. Baselines on base model M
```powershell
python -m evaluation.run_eval --dataset jbypass --tag M
python -m evaluation.run_eval --dataset paper_probe_swap --tag M --fields prompt
python -m evaluation.run_mechanism --tag M
python -m evaluation.run_perplexity --tag M
```
Quick first try: add `--limit 10 --conditions none,bridge`.

### `evaluation.run_eval`: accuracy under each ablation
Prints and saves a table of accuracy for each condition and prompt type. Output: `<out_dir>/eval_<tag>_<dataset>_<split or all>.json`.
| flag | default | what it does |
|---|---|---|
| `--dataset <name>` | `jbypass` | `jbypass` (our 1032 items), `paper_probe_swap` or `paper_multihop` (Anthropic's sets). |
| `--conditions <a,b,...>` | all ten (the two clamp ones need section 3b) | Comma-separated ablations to run, with no spaces (see the table below). |
| `--fields <a,b,...>` | `prompt,hop2_prompt,hop1_prompt` | Which prompt types to score. `prompt` is the two-hop question (main metric). `hop2_prompt` gives the bridge and asks for the answer (fair one-hop control). `hop1_prompt` asks for the bridge itself. Anthropic's datasets only have `prompt`, so pass `--fields prompt` for them. |
| `--all_items` | off | Keeps items that M gets wrong with no ablation. By default those are dropped, so accuracy measures what the ablation breaks. |

It also takes the shared, eval and split flags.

**Ablation conditions** (`--conditions` values):
| name | what it removes |
|---|---|
| `none` | nothing (clean accuracy) |
| `bridge` | the J-lens vectors of the item's bridge, over the ablation band (the intervention) |
| `wrong_bridge` | the same, for a different bridge of the same kind (checks that the ablation is specific) |
| `bridge_clamp` | lowers the bridge's J-lens readout over the band to its normal level (mean + `clamp_z` x std from section 3b) only where it is higher; other positions are untouched |
| `wrong_bridge_clamp` | the same clamp, for a different bridge of the same kind (specificity control for the clamp) |
| `random_tok` | J-lens vectors of the same number of random word tokens |
| `random_dir` | the same number of random Gaussian directions |
| `topk_light`, `topk_medium`, `topk_heavy` | the paper's ablation: the top 10 active J-lens vectors at each position, over the first third, two thirds, or all of the band |

### `evaluation.show_table`: reprint saved eval tables
```powershell
python -m evaluation.show_table results/baseline/eval_M_jbypass_all.json "results/exp 1/eval_mexp_*.json"
```
Prints the exact-match and next-token tables of saved `run_eval` results, rebuilt from their per-item records, without loading a model. Takes one or more files or glob patterns (quote patterns; the script expands them). No other flags. Writes nothing.

### `evaluation.run_mechanism`: how the model gets around the ablation
For each condition it saves the bridge's J-lens rank at each layer and position, plus the accuracy of linear probes that predict the bridge from the residual stream. Output: `<out_dir>/mechanism_<tag>_<dataset>_<split or all>.json`.
| flag | default | what it does |
|---|---|---|
| `--dataset <name>` | `jbypass` | As in `run_eval`. |
| `--conditions <a,b,...>` | `none,bridge` | Conditions to analyze under. |
| `--extra_lens <label>=<path>` | none | Also runs the analysis with another lens and saves it under `<label>`, next to the original lens's results. You can repeat it, e.g. `--extra_lens recomputed=lenses/mexp.pt`. |

It also takes the shared, eval and split flags.

### `evaluation.run_perplexity`: did fine-tuning damage general language modeling?
Perplexity on held-out WikiText. Output: `<out_dir>/perplexity_<tag>.json`. Takes only the shared and eval flags. `--limit` sets the number of texts, and `--lens_path` is accepted but ignored.

## 5. Train
**Set `train.steps` in `configs/run.yaml` first.** It is 4 (smoke-test length). Steps, batch size, learning rate, LoRA rank and seed all come from that file. To try settings without editing it, copy it and pass `--config <copy>`.
```powershell
python -m finetune.run_train --condition bridge     --split_file results/splits.json --out runs/mexp    # Mexp
python -m finetune.run_train --condition none       --split_file results/splits.json --out runs/mctrl   # Mctrl
python -m finetune.run_train --condition random_tok --split_file results/splits.json --out runs/mrand   # Mrand
python -m finetune.run_train --condition bridge_clamp --split_file results/splits.json --out runs/mclamp # clamp experiment (needs section 3b)
```
`finetune.run_train` trains a LoRA adapter on the `train` split, with the chosen ablation applied during every forward pass.
| flag | default | what it does |
|---|---|---|
| `--condition <name>` | required | The ablation applied during training: `bridge` (Mexp), `none` (Mctrl), `random_tok` or `random_dir` (Mrand), `bridge_clamp` (bridge readout capped at its normal level instead of zeroed; needs section 3b). |
| `--split_file <path>` | required | Split file from `make_splits`. Training always uses its `train` split. |
| `--out <dir>` | required | Folder to save the adapter in, along with `train_info.json` (condition, seeds, stats file used, `clamp_z`, `band` layers, per-step losses). Pass this folder to the eval commands as `--adapter`. |
| `--lens_path <file.pt>` | the profile's prefit lens | The lens whose vectors are ablated during training. |
| `--stats_path <file.pt>` | `<readout_stats_dir>/<model>.pt` if it exists | Reference readout statistics for `bridge_clamp`. |

It also takes the shared flags. It prints the loss at every step and the peak GPU memory at the end.

## 6. Evaluate fine-tuned models
Same commands as section 4, plus `--adapter`, the split flags, and a `--tag` naming the model. Run each one for every adapter (`mexp`, `mctrl`, `mrand`) and for both `test_seen` and `test_unseen`:
```powershell
python -m evaluation.run_eval       --adapter runs/mexp --split_file results/splits.json --split test_seen --tag mexp --out_dir "results/exp 1"
python -m evaluation.run_mechanism  --adapter runs/mexp --split_file results/splits.json --split test_seen --tag mexp --out_dir "results/exp 1"
python -m evaluation.run_perplexity --adapter runs/mexp --tag mexp --out_dir "results/exp 1"
```
The split name is already in the output filename, so `--tag mexp` is enough for both splits. `run_perplexity` has no split, so run it once per adapter.

## 7. Recompute the lens after training
The refit lenses for exp 2 and exp 3 (shown by the viewer, section 8); run them one at a time, since each needs the whole GPU:
```powershell
python -m jspace.fit_lens --adapter runs/mclamp       --n_prompts 100 --dim_batch 8 --out lenses/mclamp.pt        # exp 2 adapter
python -m jspace.fit_lens --adapter runs/mclamp-L8-26 --n_prompts 100 --dim_batch 8 --out lenses/mclamp-L8-26.pt  # exp 3 adapter
python -m jspace.fit_lens                             --n_prompts 100 --dim_batch 8 --out lenses/qwen3-1.7b.pt    # base M, control
```
`jspace.fit_lens` fits a new J-lens with Anthropic's `jlens.fit` on WikiText prompts, the same corpus the prefit lens used, with the adapter active (not merged). The band does not affect a fit, so no `--config` is needed. The base-M fit is a control: it uses the same recipe as the adapter fits, while the prefit Neuronpedia lens used a different one (466 prompts with early stopping, longer prompts; `jlens.fit` has no early stopping), so compare an adapter's refit with the base-M refit, not only with the prefit lens.

Cost on this 8 GB laptop GPU at `--dim_batch 8`: 53-56 s per prompt with nothing else on the GPU, about 65 s with a game open (the exp 2 fit took 1 h 37 min for 100 prompts). `--dim_batch 16` exceeds dedicated GPU memory: the Windows driver spills into shared memory instead of failing, and no prompt finished in 15 minutes.

The fit writes a checkpoint `<out>.ckpt` every 10 prompts (`fit_checkpoint_every` in a `jspace:` section; `null` = only at the end) and resumes from it: rerun an interrupted fit with the same command. Rerunning the same `--out` with a larger `--n_prompts` extends the fit (e.g. 100 -> 200 prompts costs only the new 100). The checkpoint does not record the model, so a different model or adapter needs a different `--out` (debt 003).
| flag | default | what it does |
|---|---|---|
| `--out <path>` | required | Where to save the lens `.pt`. A checkpoint `<out>.ckpt` is written while it runs and resumed from. |
| `--adapter <dir>` | none (base M) | Fits the lens on the fine-tuned model instead of M. |
| `--n_prompts <N>` | `1000` | Number of WikiText prompts to fit on. Fewer is faster and rougher. |
| `--dim_batch <N>` | `32` | Number of dimensions `jlens.fit` processes at once. Lower it if you run out of GPU memory (8 on an 8 GB GPU). |

It also takes the shared flags.

Then compare the new lens with the original side by side, or ablate with it:
```powershell
python -m evaluation.run_mechanism --adapter runs/mexp --extra_lens recomputed=lenses/mexp.pt --split_file results/splits.json --split test_seen --tag mexp
python -m evaluation.run_eval      --adapter runs/mexp --lens_path lenses/mexp.pt --split_file results/splits.json --split test_seen --tag mexp_relens
```
The second command uses a new tag (`mexp_relens`) so it doesn't overwrite the results from section 6.

## 8. Visualize
```powershell
python -m viz.run_app                                   # then open http://127.0.0.1:7860
python -m viz.run_app --config configs/exp3.yaml        # exp 3 band (layers 8-26)
python -m viz.run_app --device cpu                      # no GPU (slow)
```
`viz.run_app` starts a local Gradio app for looking at one example at a time: it generates under one ablation and shows the J-lens readout of every layer over the prompt and the response (jlens's heatmap page), through the original lens and through the model's refit lens. The page opens at once without loading a model (startup loads only the tokenizer); a model loads on the first **Run** (a few seconds on GPU). The first run in each process downloads d3 from a CDN, so it needs internet once. An env created before gradio was added needs `pip install gradio==6.32.0` and `pip install -e .` (a fresh `conda env create -f environment.yml` already has it). At startup it prints `CC-CEDICT: <N> entries, <M> headwords <- <path>` (or `Translations off (dictionary_file: null)`); it stops with an error naming the download page if the dictionary file from section 1 is missing.
| flag | default | what it does |
|---|---|---|
| `--config <path>` | `configs/run.yaml` | Run settings; `generation.max_new_tokens` is the response length, and its `jspace:` section sets the band and readout statistics (e.g. `configs/exp3.yaml` for layers 8-26). |
| `--model <name>` | `model:` in run.yaml | The base model of this viewer: the model dropdown lists it and its adapters only. Must have a prefit lens (`qwen3-1.7b`). |
| `--device <name>` | the profile's `device` | Where every model choice runs. |
| `--viz_config <path>` | `configs/viz.yaml` | Viewer settings: adapters folder, refit-lens folder (`lenses_dir`), split file, top-n, tracked-token cap, iframe height, host and port. |

**Which `--config` for which adapter:** the ablations use the band of the viewer's `--config`, so open `runs/mclamp` (exp 2) with the default `configs/run.yaml` (band 11-26) and `runs/mclamp-L8-26` (exp 3) with `--config configs/exp3.yaml` (band 8-26). Both adapters are listed under either config; with ablation `none` the band does not matter.

What you can do in the UI, top to bottom:
- **Dataset / Split**, then **Example** (or **Random example**): splits come from `results/splits_expanded.json` (`all` = no filter); only items whose bridge the lens can read are listed.
- **Prompt field / Ablation**: only the fields the item has and the conditions that can be built for it (`wrong_bridge` / `wrong_bridge_clamp` need a swap bridge; the clamp conditions appear only when `<readout_stats_dir>/<model>.pt` of `--config` exists). `none` is the clean run. The item card below shows the prompt, bridge and expected answers.
- **Model**, then **Run**: the base model or any adapter in `runs/` trained from it (only one model is in memory; running another model reloads). Run generates and shows the response, whether it is correct, and the heatmaps.
- **Result cache**: every run is kept for the selected question, so flipping the model dropdown back to a model that already ran shows its result at once with no reload (in either tab); a model not yet run shows "Not run yet ... Press Run.". Changing the question clears the result on screen, and the first Run on a new question drops the old question's results. The table above the summary lists every run of the question (model, prompt field, ablation, response, correct).
- **Two lens tabs**: "Original lens" reads the model through the base model's prefit lens; "Refit lens (this model)" through the model's own refit lens: `lenses/<adapter folder name>.pt`, or `lenses/<profile>.pt` for the base model (section 7). Generation and the ablations always use the original lens; the refit lens only reads out. Without a refit lens the tab shows the command that fits one. Lenses are found at startup, so **restart the viewer after fitting a lens**.
- **Heatmap**: click a cell for its top tokens, click a pin to see its rank over every (position, layer); the bridge and the answers are pinned. Rows at ablated layers show the residual after the ablation.
- **Translations**: click a Chinese token (in the By Layer / By Pos lists, a pinned chip, the prompt strip or a grid cell) for its translation: a box at the bottom right shows the pinyin and senses of each word (CC-CEDICT). Clicking still pins the token as before. Other scripts show "no dictionary entry"; English tokens close the box; × closes it.

## 9. Experiment 3: clamp from layer 8
Exp 2 with the band widened from layers 11-26 (the default fractions) to layers 8-26. Everything else is unchanged: `configs/exp3.yaml` is `configs/run.yaml` (same model, LoRA, 500 steps, seed) plus `jspace: {band_layers: [8, 26], readout_stats_dir: ../stats/exp3}`. Pass `--config configs/exp3.yaml` to every command so training, statistics, evals and the viewer all use the same band.
```powershell
# readout statistics over the whole dataset for layers 8-26 (exp 2's stats/qwen3-1.7b.pt: whole dataset, 1026 texts) -> stats/exp3/qwen3-1.7b.pt
python -m evaluation.run_readout_stats --config configs/exp3.yaml
# Mclamp with the wider band (bridge_clamp, clamp_z 0.5, train split of splits_expanded, as runs/mclamp)
python -m finetune.run_train --config configs/exp3.yaml --condition bridge_clamp --split_file results/splits_expanded.json --out runs/mclamp-L8-26
# base M under the wider band (exp 2: tags M_clamp at clamp_z 2.0 and M_z05 at 0.5)
python -m evaluation.run_eval --config configs/exp3.yaml --split_file results/splits_expanded.json --split test_unseen --conditions none,bridge,bridge_clamp,wrong_bridge_clamp --all_items --tag M_L8-26 --out_dir "results/exp 3"
# the adapter: four diagnostic conditions, then all ten (exp 2: tags mclamp-diag, mclamp-all)
python -m evaluation.run_eval --config configs/exp3.yaml --adapter runs/mclamp-L8-26 --split_file results/splits_expanded.json --split test_unseen --conditions none,bridge_clamp,bridge,wrong_bridge_clamp --all_items --tag mclamp-L8-26-diag --out_dir "results/exp 3"
python -m evaluation.run_eval --config configs/exp3.yaml --adapter runs/mclamp-L8-26 --split_file results/splits_expanded.json --split test_unseen --all_items --tag mclamp-L8-26-all --out_dir "results/exp 3"
# viewer on the same band (clamp conditions use stats/exp3/)
python -m viz.run_app --config configs/exp3.yaml
```
All evals use all three prompt types (the default `--fields`) on the 154 `test_unseen` items, as in exp 2. `--all_items` keeps items the adapter gets wrong clean (exp 2's adapter runs used it; for base M it changes nothing, since the splits hold only items M answers). The diag run's four conditions are a subset of the all run, kept to mirror exp 2. `train_info.json` and every eval JSON record the band (`"band": [8, ..., 26]`); check it before comparing with `results/exp 2/` (`[11, ..., 26]`). To repeat exp 2's clamp_z sweep on this band, copy `configs/exp3.yaml` and add `clamp_z: 2.0` under `jspace:` (statistics do not depend on `clamp_z`, so the same `readout_stats_dir` works).

## Outputs
| command | writes |
|---|---|
| `make_splits` | `results/splits.json` (or `--out`) |
| `run_readout_stats` | `<readout_stats_dir>/<model>.pt`, default `stats/<model>.pt` (or `--out`) |
| `run_eval` | `<out_dir>/eval_<tag>_<dataset>_<split or all>.json` (`--out_dir`, default `results`) |
| `run_mechanism` | `<out_dir>/mechanism_<tag>_<dataset>_<split or all>.json` |
| `run_perplexity` | `<out_dir>/perplexity_<tag>.json` |
| `run_lens_checks` | `<out_dir>/lens_checks_<tag>.json` |
| `run_train` | `<out>/` (adapter + `train_info.json`) |
| `fit_lens` | `<out>` (lens `.pt`) and `<out>.ckpt` (checkpoint, resumed by a rerun with the same `--out`) |
| `viz.run_app` | nothing (serves a local web page) |
