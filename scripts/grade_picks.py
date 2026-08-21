from __future__ import annotations

from pathlib import Path
from datetime import datetime, timedelta
import requests
import pandas as pd


# Anchor to the project root (parent of this scripts/ folder) so output always
# resolves to the same place, regardless of the working directory the script
# was launched from (module invocation, cron, or run manually from scripts/).
BASE_DIR = Path(__file__).resolve().parent.parent
INPUT_DIR = BASE_DIR / "output"
OUTPUT_DIR = BASE_DIR / "output"
API_BASE = "https://statsapi.mlb.com/api/v1"


def yesterday_str() -> str:
    return (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")


def input_file_for(date_str: str) -> Path:
    return INPUT_DIR / f"hit_candidates_{date_str}.csv"


def graded_file_for(date_str: str) -> Path:
    return OUTPUT_DIR / f"graded_picks_{date_str}.csv"


def summary_file_for(date_str: str) -> Path:
    return OUTPUT_DIR / f"qa_summary_{date_str}.csv"


def fetch_boxscore(game_pk: int) -> dict:
    url = f"{API_BASE}/game/{game_pk}/boxscore"
    resp = requests.get(url, timeout=30)
    resp.raise_for_status()
    return resp.json()


def safe_int(value, default: int = 0) -> int:
    try:
        if value is None or value == "":
            return default
        return int(value)
    except (TypeError, ValueError):
        return default


def extract_batter_stat_rows(boxscore: dict) -> pd.DataFrame:
    rows = []
    teams = boxscore.get("teams", {})

    for side in ("home", "away"):
        team_block = teams.get(side, {})
        team_name = team_block.get("team", {}).get("name")
        players = team_block.get("players", {})

        for _, pdata in players.items():
            person = pdata.get("person", {})
            stats = pdata.get("stats", {}).get("batting", {}) or {}

            if not person:
                continue

            player_id = safe_int(person.get("id"), default=-1)
            full_name = person.get("fullName")

            hits = safe_int(stats.get("hits"))
            doubles = safe_int(stats.get("doubles"))
            triples = safe_int(stats.get("triples"))
            home_runs = safe_int(stats.get("homeRuns"))
            at_bats = safe_int(stats.get("atBats"))
            plate_appearances = safe_int(stats.get("plateAppearances"))
            runs = safe_int(stats.get("runs"))
            rbi = safe_int(stats.get("rbi"))
            walks = safe_int(stats.get("baseOnBalls"))
            strikeouts = safe_int(stats.get("strikeOuts"))

            singles = max(hits - doubles - triples - home_runs, 0)
            xbh = doubles + triples + home_runs
            had_hit = int(hits > 0)

            if hits == 0:
                result_label = "no_hit"
            elif hits >= 2 and xbh >= 1:
                result_label = "multi_hit_xbh"
            elif hits >= 2:
                result_label = "multi_hit"
            elif home_runs >= 1:
                result_label = "home_run"
            elif triples >= 1:
                result_label = "triple"
            elif doubles >= 1:
                result_label = "double"
            else:
                result_label = "single_only"

            rows.append(
                {
                    "batter_id": player_id,
                    "batter_api_name": full_name,
                    "team_name_api": team_name,
                    "had_hit": had_hit,
                    "hit_count": hits,
                    "single_count": singles,
                    "double_count": doubles,
                    "triple_count": triples,
                    "home_run_count": home_runs,
                    "xbh_count": xbh,
                    "at_bats": at_bats,
                    "plate_appearances": plate_appearances,
                    "runs_scored": runs,
                    "rbi": rbi,
                    "walks": walks,
                    "strikeouts": strikeouts,
                    "result_label": result_label,
                }
            )

    return pd.DataFrame(rows)


def grade_picks(picks: pd.DataFrame) -> pd.DataFrame:
    if picks.empty:
        return picks.copy()

    if "game_pk" not in picks.columns:
        raise ValueError("Missing required column: game_pk")

    if "batter_id" not in picks.columns:
        raise ValueError("Missing required column: batter_id")

    game_pks = (
        picks["game_pk"]
        .dropna()
        .astype(int)
        .drop_duplicates()
        .tolist()
    )

    all_stats = []

    for game_pk in game_pks:
        try:
            boxscore = fetch_boxscore(game_pk)
            game_stats = extract_batter_stat_rows(boxscore)
            game_stats["game_pk"] = game_pk
            all_stats.append(game_stats)
        except Exception as exc:
            all_stats.append(
                pd.DataFrame(
                    [
                        {
                            "game_pk": game_pk,
                            "batter_id": -1,
                            "batter_api_name": None,
                            "team_name_api": None,
                            "had_hit": None,
                            "hit_count": None,
                            "single_count": None,
                            "double_count": None,
                            "triple_count": None,
                            "home_run_count": None,
                            "xbh_count": None,
                            "at_bats": None,
                            "plate_appearances": None,
                            "runs_scored": None,
                            "rbi": None,
                            "walks": None,
                            "strikeouts": None,
                            "result_label": "boxscore_fetch_failed",
                            "qa_error": str(exc),
                        }
                    ]
                )
            )

    stats_df = pd.concat(all_stats, ignore_index=True) if all_stats else pd.DataFrame()

    graded = picks.copy()
    graded["game_pk"] = graded["game_pk"].astype(int)
    graded["batter_id"] = graded["batter_id"].astype(int)

    merge_cols = [
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
    ]
    if "qa_error" in stats_df.columns:
        merge_cols.append("qa_error")

    graded = graded.merge(
        stats_df[merge_cols],
        on=["game_pk", "batter_id"],
        how="left",
    )

    graded["graded_at"] = datetime.utcnow().isoformat(timespec="seconds") + "Z"
    graded["had_hit"] = graded["had_hit"].fillna(0).astype(int)

    for col in [
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
    ]:
        if col in graded.columns:
            graded[col] = graded[col].fillna(0).astype(int)

    graded["result_label"] = graded["result_label"].fillna("not_found_in_boxscore")

    return graded


def build_summary(graded: pd.DataFrame, date_str: str) -> pd.DataFrame:
    total_picks = len(graded)
    hits = int(graded["had_hit"].sum()) if "had_hit" in graded.columns else 0
    singles = int(graded["single_count"].sum()) if "single_count" in graded.columns else 0
    doubles = int(graded["double_count"].sum()) if "double_count" in graded.columns else 0
    triples = int(graded["triple_count"].sum()) if "triple_count" in graded.columns else 0
    home_runs = int(graded["home_run_count"].sum()) if "home_run_count" in graded.columns else 0
    xbh = int(graded["xbh_count"].sum()) if "xbh_count" in graded.columns else 0

    hit_rate = round(hits / total_picks, 4) if total_picks else 0.0
    xbh_rate = round(xbh / total_picks, 4) if total_picks else 0.0

    summary = pd.DataFrame(
        [
            {
                "date": date_str,
                "total_picks": total_picks,
                "picks_with_hit": hits,
                "hit_rate": hit_rate,
                "total_singles": singles,
                "total_doubles": doubles,
                "total_triples": triples,
                "total_home_runs": home_runs,
                "total_xbh": xbh,
                "xbh_rate": xbh_rate,
                "graded_at": datetime.utcnow().isoformat(timespec="seconds") + "Z",
            }
        ]
    )

    return summary


def main():
    date_str = yesterday_str()
    in_file = input_file_for(date_str)

    if not in_file.exists():
        raise FileNotFoundError(f"Input file not found: {in_file}")

    picks = pd.read_csv(in_file)

    required = ["game_pk", "batter_id"]
    missing = [c for c in required if c not in picks.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    graded = grade_picks(picks)
    summary = build_summary(graded, date_str)

    out_file = graded_file_for(date_str)
    qa_file = summary_file_for(date_str)

    graded.to_csv(out_file, index=False)
    summary.to_csv(qa_file, index=False)

    print(f"Wrote {out_file}")
    print(f"Wrote {qa_file}")


if __name__ == "__main__":
    main()