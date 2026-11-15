"""Settings for the evaluation suite. Change values here, not inline."""

from dataclasses import dataclass, field
from pathlib import Path

DATA_DIR = Path(__file__).parent / "data"


# TODO(debt-002): dataclass defaults, while run and model settings are YAML.
@dataclass(frozen=True)
class EvalConfig:
    # Datasets (name -> path). paper_* are Anthropic's, copied from jacobian-lens.
    datasets: dict = field(default_factory=lambda: {
        "jbypass": DATA_DIR / "jbypass_two_hop.json",
        "paper_probe_swap": DATA_DIR / "probe-swap.json",
        "paper_multihop": DATA_DIR / "lens-eval-multihop.json",
    })
    # Ablation conditions to run (names registered in evaluation/conditions.py).
    conditions: tuple = ("none", "bridge", "wrong_bridge", "random_tok", "random_dir",
                         "topk_light", "topk_medium", "topk_heavy")
    # Question types: prompt fields on an item (two-hop, plus one-hop controls).
    prompt_fields: tuple = ("prompt", "hop2_prompt", "hop1_prompt")
    seed: int = 0
    # Greedy tokens generated per answer for exact-match scoring.
    max_answer_tokens: int = 5
    # Readout check: rank threshold for "bridge is readable" (pass@k in the paper).
    readout_k: int = 10
    # Mechanism analysis: conditions to read out / probe under.
    mechanism_conditions: tuple = ("none", "bridge")
    # Linear probes: cross-validation folds (bridges need >= this many items),
    # optimizer steps, L2 weight decay.
    probe_folds: int = 2
    probe_steps: int = 300
    probe_weight_decay: float = 1e-3
    # Perplexity sanity check on held-out WikiText (validation split, not the
    # train split the lens was fit on).
    ppl_n_texts: int = 200
    ppl_max_len: int = 128
    ppl_min_chars: int = 600
    ppl_batch_size: int = 8         # texts per forward pass; lower it on a small GPU
    out_dir: Path = Path("results")
