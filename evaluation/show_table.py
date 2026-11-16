"""Print the accuracy tables of saved run_eval results, without loading a model.

python -m evaluation.show_table results/eval_M_jbypass_all.json
python -m evaluation.show_table "results/eval_mexp_*.json" results/eval_M_jbypass_all.json
"""

import argparse
import glob
import json
from pathlib import Path

from evaluation.runner import accuracy_table, format_table


def expand(patterns) -> list[Path]:
    """Each path or glob pattern -> its sorted matches, in argument order. Raises if one matches nothing."""
    paths = []
    for pattern in patterns:
        matches = sorted(glob.glob(pattern))
        if not matches:
            raise FileNotFoundError(f"no file matches {pattern!r}")
        paths.extend(Path(m) for m in matches)
    return paths


def load_result(path: Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def describe(path: Path, result: dict) -> str:
    """Header plus both tables, rebuilt from the records (the stored tables are ignored)."""
    records = result["records"]
    conditions = list(dict.fromkeys(r["condition"] for r in records))
    n_items = len({r["name"] for r in records})
    adapter = result["adapter"] or "base model"
    header = (f"{Path(path).name}: tag={result['tag']} dataset={result['dataset']} "
              f"split={result['split']} adapter={adapter} items={n_items}")
    return "\n".join([
        header,
        "exact match (primary):\n" + format_table(accuracy_table(records), conditions),
        "next token (paper's metric):\n"
        + format_table(accuracy_table(records, key="next_token_correct"), conditions),
    ])


def main():
    p = argparse.ArgumentParser()
    p.add_argument("files", nargs="+", help="result files or glob patterns (expanded here, quote them)")
    args = p.parse_args()
    print("\n\n".join(describe(path, load_result(path)) for path in expand(args.files)))


if __name__ == "__main__":
    main()
