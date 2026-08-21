from __future__ import annotations

import argparse
from pathlib import Path
import pandas as pd


BASE_DIR = Path(__file__).resolve().parents[1]
OUTPUT_DIR = BASE_DIR / "output"


def resolve_input_path(raw: str) -> Path:
    p = Path(raw)

    if p.exists():
        return p.resolve()

    candidates = [
        BASE_DIR / raw,
        OUTPUT_DIR / raw,
        Path.cwd() / raw,
    ]

    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()

    checked = "\n".join(str(c) for c in candidates)
    raise FileNotFoundError(
        f"File not found: {raw}\nChecked:\n{checked}"
    )


def load_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    return pd.read_csv(path)


def combine_history(files: list[Path]) -> pd.DataFrame:
    frames = []
    for path in files:
        df = load_csv(path)
        frames.append(df)

    if not frames:
        raise ValueError("No graded pick files provided.")

    combined = pd.concat(frames, ignore_index=True)

    required = ["date", "game_pk", "batter_id"]
    missing = [c for c in required if c not in combined.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    combined = combined.drop_duplicates(
        subset=["date", "game_pk", "batter_id"],
        keep="last",
    ).copy()

    combined = combined.sort_values(["date", "game_pk", "batter_id"]).reset_index(drop=True)
    return combined


def main():
    parser = argparse.ArgumentParser(
        description="Combine daily graded_picks CSV files into output/graded_history.csv"
    )
    parser.add_argument(
        "files",
        nargs="+",
        help="One or more graded_picks CSV files",
    )
    parser.add_argument(
        "--out",
        default=str(OUTPUT_DIR / "graded_history.csv"),
        help="Output history CSV path",
    )
    parser.add_argument(
        "--fresh",
        action="store_true",
        help="Ignore any existing --out file and rebuild history from just the given files "
        "(default behavior appends/dedupes onto the existing --out file).",
    )
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    out_path = Path(args.out)
    if not out_path.is_absolute():
        out_path = (BASE_DIR / out_path).resolve()

    files = [resolve_input_path(f) for f in args.files]

    # Merge onto whatever is already in the history file, so daily runs that
    # only pass in the latest day's graded_picks CSV accumulate over time
    # instead of clobbering everything that came before.
    if out_path.exists() and not args.fresh:
        files = [out_path] + files

    combined = combine_history(files)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(out_path, index=False)

    print(f"Wrote {out_path}")
    print(f"Rows: {len(combined)}")


if __name__ == "__main__":
    main()