"""Fetch a profile's model weights and tokenizer into the default Hugging Face cache.

Usage:  python scripts/download_model.py [--model qwen3-0.6b] [--config configs/run.yaml]
"""

from huggingface_hub import snapshot_download

from _cli import parse_run_args, resolve_run


def main() -> None:
    _, profile = resolve_run(parse_run_args(__doc__))
    path = snapshot_download(repo_id=profile.hf_id)
    print(f"{profile.name}: {profile.hf_id} -> {path}")


if __name__ == "__main__":
    main()
