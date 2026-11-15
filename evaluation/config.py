"""Settings for the evaluation suite. Change values here, not inline."""

from dataclasses import dataclass, field
from pathlib import Path

DATA_DIR = Path(__file__).parent / "data"


@dataclass(frozen=True)
class EvalConfig:
    # Datasets (name -> path). paper_* are Anthropic's, copied from jacobian-lens.
    datasets: dict = field(default_factory=lambda: {
        "jbypass": DATA_DIR / "jbypass_two_hop.json",
        "paper_probe_swap": DATA_DIR / "probe-swap.json",
        "paper_multihop": DATA_DIR / "lens-eval-multihop.json",
    })
    # Ablation conditions to run (names registered in evaluation/conditions.py).
    conditions: tuple = ("none", "bridge", "wrong_bridge", "random_tok", "random_dir", "topk")
    # Question types: prompt fields on an item (two-hop, plus one-hop controls).
    prompt_fields: tuple = ("prompt", "hop2_prompt", "hop1_prompt")
    seed: int = 0
    # Readout check: rank threshold for "bridge is readable" (pass@k in the paper).
    readout_k: int = 10
    out_dir: Path = Path("results")
