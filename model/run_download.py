"""Fetch a profile's model weights and tokenizer into the default Hugging Face cache.

Usage:  python -m model.run_download [--model qwen3-0.6b] [--config configs/run.yaml]
"""

import argparse

from huggingface_hub import snapshot_download

from model.cli import add_profile_args, resolve_profile


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    add_profile_args(parser)
    _, profile = resolve_profile(parser.parse_args())
    path = snapshot_download(repo_id=profile.hf_id)
    print(f"{profile.name}: {profile.hf_id} -> {path}")


if __name__ == "__main__":
    main()
