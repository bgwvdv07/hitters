from __future__ import annotations

import argparse
import datetime as dt
import logging
from pathlib import Path

import pandas as pd


# Anchor to the project root (parent of this scripts/ folder) so output always
# resolves to the same place, regardless of the working directory the script
# was launched from (module invocation, cron, or run manually from scripts/).
BASE_DIR = Path(__file__).resolve().parent.parent
OUTPUT_DIR = BASE_DIR / "output"
MASTER_FILE = OUTPUT_DIR / "training_history.csv"

logger = logging.getLogger("update_training_history")


def setup_logging(verbosity: int) -> None:
    if verbosity <= 0:
        level = logging.WARNING
    elif verbosity == 1:
        level = logging.INFO
    else:
        level = logging.DEBUG

    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S",
    )


def yesterday_str() -> str:
    return (dt.datetime.now() - dt.timedelta(days=1)).strftime("%Y-%m-%d")


def candidates_file_for(date_str: str) -> Path:
    return OUTPUT_DIR / f"hit_candidates_{date_str}.csv"


def graded_file_for(date_str: str) -> Path:
    return OUTPUT_DIR / f"graded_picks_{date_str}.csv"


def training_rows_file_for(date_str: str) -> Path:
    return OUTPUT_DIR / f"training_rows_{date_str}.csv"


def history_summary_file() -> Path:
    return OUTPUT_DIR / "training_history_summary.csv"


def history_by_date_file() -> Path:
    return OUTPUT_DIR / "training_history_by_date.csv"


def read_csv_required(path: Path, label: str) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"{label} not found: {path}")
    logger.info("Reading %s", path)
    return pd.read_csv(path)


def normalize_keys(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    if "game_pk" in out.columns:
        out["game_pk"] = pd.to_numeric(out["game_pk"], errors="coerce").astype("Int64")

    if "batter_id" in out.columns:
        out["batter_id"] = pd.to_numeric(out["batter_id"], errors="coerce").astype("Int64")

    if "date" in out.columns:
        out["date"] = out["date"].astype(str)

    return out


def validate_required_columns(df: pd.DataFrame, df_name: str, required: list[str]) -> None:
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"{df_name} missing required columns: {missing}")


def validate_unique_keys(df: pd.DataFrame, df_name: str, keys: list[str]) -> None:
    dupes = df[df.duplicated(subset=keys, keep=False)].sort_values(keys)
    if not dupes.empty:
        sample = dupes[keys].head(10).to_dict(orient="records")
        raise ValueError(f"{df_name} has duplicate key rows on {keys}. Sample: {sample}")


def choose_dedupe_keys(df: pd.DataFrame) -> list[str]:
    if all(c in df.columns for c in ["date", "game_pk", "batter_id"]):
        return ["date", "game_pk", "batter_id"]
    if all(c in df.columns for c in ["game_pk", "batter_id"]):
        return ["game_pk", "batter_id"]
    raise ValueError("Could not find valid dedupe keys in combined history.")


def merge_daily_predictions_and_outcomes(
    candidates: pd.DataFrame,
    graded: pd.DataFrame,
    include_ungraded: bool = True,
) -> pd.DataFrame:
    keys = ["game_pk", "batter_id"]

    candidates = normalize_keys(candidates)
    graded = normalize_keys(graded)

    validate_required_columns(candidates, "candidates", keys)
    validate_required_columns(graded, "graded", keys)

    validate_unique_keys(candidates, "candidates", keys)
    validate_unique_keys(graded, "graded", keys)

    graded_outcome_cols = [
        "game_pk",
        "batter_id",
        "batter_api_name",
        "team_name_api",
        "had_hit",
        "hit_count",
        "single_count",
        "double_count",
        "triple_count",
        "home_run_count",
        "xbh_count",
        "at_bats",
        "plate_appearances",
        "runs_scored",
        "rbi",
        "walks",
        "strikeouts",
        "result_label",
        "qa_error",
        "graded_at",
    ]
    graded_outcome_cols = [c for c in graded_outcome_cols if c in graded.columns]

    merged = candidates.merge(
        graded[graded_outcome_cols],
        on=keys,
        how="left",
        validate="one_to_one",
    )

    merged["training_row_created_at"] = dt.datetime.utcnow().isoformat(timespec="seconds") + "Z"

    if "had_hit" in merged.columns:
        merged["had_hit"] = pd.to_numeric(merged["had_hit"], errors="coerce").fillna(0).astype(int)

    numeric_outcomes = [
        "hit_count",
        "single_count",
        "double_count",
        "triple_count",
        "home_run_count",
        "xbh_count",
        "at_bats",
        "plate_appearances",
        "runs_scored",
        "rbi",
        "walks",
        "strikeouts",
    ]
    for col in numeric_outcomes:
        if col in merged.columns:
            merged[col] = pd.to_numeric(merged[col], errors="coerce").fillna(0).astype(int)

    if "result_label" in merged.columns:
        merged["result_label"] = merged["result_label"].fillna("not_graded")

    if not include_ungraded and "result_label" in merged.columns:
        before = len(merged)
        merged = merged[
            ~merged["result_label"].isin(
                ["not_graded", "not_found_in_boxscore", "boxscore_fetch_failed"]
            )
        ].copy()
        logger.info("Filtered ungraded/problem rows: %s -> %s", before, len(merged))

    logger.info("Merged daily training rows: %s", len(merged))
    return merged


def append_to_master(master_path: Path, daily_rows: pd.DataFrame) -> pd.DataFrame:
    if master_path.exists():
        master = pd.read_csv(master_path)
        logger.info("Loaded existing master history: %s rows", len(master))
        combined = pd.concat([master, daily_rows], ignore_index=True, sort=False)
    else:
        logger.info("No existing master history found. Creating new file.")
        combined = daily_rows.copy()

    dedupe_keys = choose_dedupe_keys(combined)
    before = len(combined)
    combined = combined.drop_duplicates(subset=dedupe_keys, keep="last").reset_index(drop=True)
    after = len(combined)

    logger.info("Applied dedupe on %s: %s -> %s rows", dedupe_keys, before, after)

    combined.to_csv(master_path, index=False)
    return combined


def build_history_outputs(history: pd.DataFrame) -> None:
    total_rows = len(history)
    total_hits = int(history["had_hit"].sum()) if "had_hit" in history.columns else 0
    overall_hit_rate = round(total_hits / total_rows, 4) if total_rows else 0.0
    distinct_dates = int(history["date"].nunique()) if "date" in history.columns else 0

    summary = pd.DataFrame(
        [
            {
                "total_rows": total_rows,
                "total_hits": total_hits,
                "overall_hit_rate": overall_hit_rate,
                "distinct_dates": distinct_dates,
                "generated_at": dt.datetime.utcnow().isoformat(timespec="seconds") + "Z",
            }
        ]
    )
    summary.to_csv(history_summary_file(), index=False)

    if "date" in history.columns:
        by_date = (
            history.groupby("date", dropna=False)
            .agg(
                rows=("date", "size"),
                hits=("had_hit", "sum"),
            )
            .reset_index()
        )
        by_date["hit_rate"] = (by_date["hits"] / by_date["rows"]).round(4)
        by_date.to_csv(history_by_date_file(), index=False)

    logger.info("Updated history summary outputs.")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Merge daily MLB hit candidates with graded outcomes into a cumulative training history."
    )
    parser.add_argument("--date", default=yesterday_str(), help="Slate date in YYYY-MM-DD format.")
    parser.add_argument(
        "--include-ungraded",
        action="store_true",
        help="Keep rows with missing/problem grading labels in the master history.",
    )
    parser.add_argument(
        "--master-file",
        default=str(MASTER_FILE),
        help="Path to the cumulative training history CSV.",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="count",
        default=0,
        help="increase terminal verbosity (-v for progress, -vv for debug)",
    )

    args = parser.parse_args()
    setup_logging(args.verbose)

    date_str = args.date
    master_path = Path(args.master_file)

    candidates_path = candidates_file_for(date_str)
    graded_path = graded_file_for(date_str)
    daily_out_path = training_rows_file_for(date_str)

    candidates = read_csv_required(candidates_path, "Candidates file")
    graded = read_csv_required(graded_path, "Graded file")

    merged_daily = merge_daily_predictions_and_outcomes(
        candidates=candidates,
        graded=graded,
        include_ungraded=args.include_ungraded,
    )

    if merged_daily.empty:
        print(f"No training rows to write for {date_str}")
        return

    merged_daily.to_csv(daily_out_path, index=False)
    history = append_to_master(master_path, merged_daily)
    build_history_outputs(history)

    print(f"Wrote {daily_out_path}")
    print(f"Updated {master_path}")
    print(f"Wrote {history_summary_file()}")
    if "date" in history.columns:
        print(f"Wrote {history_by_date_file()}")


if __name__ == "__main__":
    main()