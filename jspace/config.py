"""Settings for loading the model and its J-lens. Change values here, not inline."""

from dataclasses import dataclass


@dataclass(frozen=True)
class JSpaceConfig:
    # Model the prefit lens was fit on (Neuronpedia config.yaml: hf_model_name).
    model_name: str = "Qwen/Qwen3-1.7B"
    dtype: str = "bfloat16"
    device: str = "cuda"

    # Prefit lens from Neuronpedia (fit with Anthropic's jlens.fit on wikitext).
    # Set lens_path to a local .pt instead to use a lens we fit ourselves
    # (e.g. one recomputed on a fine-tuned model).
    lens_repo: str = "neuronpedia/jacobian-lens"
    lens_revision: str = "main"
    lens_file: str = "qwen3-1.7b/jlens/Salesforce-wikitext/Qwen3-1.7B_jacobian_lens.pt"
    lens_path: str | None = None

    # Workspace band as a fraction of depth. The paper reports ~L38 to ~L92 on a
    # 0-100 scale for Claude; it gives no band for Qwen, so we reuse the fractions.
    band_start: float = 0.38
    band_end: float = 0.92

    # Paper's J-space ablation: top-k most active J-lens vectors per position.
    ablation_top_k: int = 10
    # Paper excludes tokens in the clean forward pass's top-k output from ablation.
    ablation_exclude_top_k: int = 10
