import argparse
import datetime as dt
from pathlib import Path
from typing import Optional
import re
import pandas as pd
import requests
from io import StringIO
import logging
import json
import os
import time
from bs4 import BeautifulSoup
import random
import pickle
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split


# Anchor to the project root (parent of this scripts/ folder) so output always
# resolves to the same place, regardless of the working directory the script
# was launched from (module invocation, cron, or run manually from scripts/).
PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = PROJECT_ROOT / "output"

BASE = "https://statsapi.mlb.com/api/v1"
WEATHER = "https://api.open-meteo.com/v1/forecast"
UA = {"User-Agent": "Mozilla/5.0"}
TEAM_CODE_MAP = {
"arizona diamondbacks": "ARI",
"atlanta braves": "ATL",
"baltimore orioles": "BAL",
"boston red sox": "BOS",
"chicago cubs": "CHC",
"chicago white sox": "CWS",
"cincinnati reds": "CIN",
"cleveland guardians": "CLE",
"colorado rockies": "COL",
"detroit tigers": "DET",
"houston astros": "HOU",
"kansas city royals": "KC",
"los angeles angels": "LAA",
"los angeles dodgers": "LAD",
"miami marlins": "MIA",
"milwaukee brewers": "MIL",
"minnesota twins": "MIN",
"new york mets": "NYM",
"new york yankees": "NYY",
"athletics": "ATH",
"philadelphia phillies": "PHI",
"pittsburgh pirates": "PIT",
"san diego padres": "SD",
"san francisco giants": "SF",
"seattle mariners": "SEA",
"st louis cardinals": "STL",
"tampa bay rays": "TB",
"texas rangers": "TEX",
"toronto blue jays": "TOR",
"washington nationals": "WSH",
}

logger = logging.getLogger("mlb_hit_finder")

# insidethepen.com uses non-standard codes for a few teams vs. the rest of this
# pipeline (which follows the standard MLB abbreviations in TEAM_CODE_MAP).
# Add entries here as other teams are discovered to differ.
INSIDETHEPEN_CODE_OVERRIDES = {
    "ARI": "AZ",
}

# A batter counts as "hot" once their current consecutive-games-with-a-hit
# streak reaches this length. Games where the batter didn't actually bat
# (0 at-bats) are skipped rather than breaking the streak.
HOT_STREAK_MIN_GAMES = 3


def setup_logging(verbosity):
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


def safe_int_or_none(value):
    try:
        if value is None or value == "":
            return None
        return int(value)
    except (TypeError, ValueError):
        return None


def safe_num(series, default=0.0):
    s = pd.to_numeric(series, errors="coerce")
    med = s.median()
    if pd.isna(med):
        med = default
    return s.fillna(med)


def safe_norm(series, default=0.0):
    s = pd.to_numeric(series, errors="coerce")
    if s.notna().sum() == 0:
        return pd.Series(default, index=series.index, dtype="float64")
    med = s.median()
    if pd.isna(med):
        med = default
    s = s.fillna(med)
    return (s - s.min()) / (s.max() - s.min() + 1e-9)


def norm(series):
    s = pd.to_numeric(series, errors="coerce")
    lo, hi = s.min(), s.max()
    if pd.isna(lo) or pd.isna(hi) or hi == lo:
        return pd.Series([0.5] * len(s), index=s.index)
    return (s - lo) / (hi - lo)


def save_map(path, data):
    with open(path, "w") as f:
        json.dump(data, f)


def load_map_if_fresh(path, max_age_hours):
    if not os.path.exists(path):
        return None
    age_hours = (time.time() - os.path.getmtime(path)) / 3600
    if age_hours > max_age_hours:
        return None
    with open(path) as f:
        return json.load(f)


def clean_key(x):
    if pd.isna(x):
        return None
    x = str(x).strip().lower()
    x = x.replace(".", "")
    x = x.replace(",", "")
    x = " ".join(x.split())
    return x


def fatigue_to_risk(score):
    if score >= 7:
        return 2
    elif score >= 4:
        return 1
    return 0


def parse_pitch_count(cell):
    if pd.isna(cell):
        return 0, False
    s = str(cell).strip()
    if not s:
        return 0, False

    matches = re.findall(r"(\d+)-(\d+)", s)
    pitches = int(matches[-1][0]) if matches else 0
    appeared = bool(re.search(r"\b\d+\.\d\b", s)) or pitches > 0
    return pitches, appeared


def fetch_bullpen_daily_stats(team_names=None):
    if not team_names:
        team_names = list(TEAM_CODE_MAP.keys())

    out = {}
    for team in team_names:
        try:
            stats = fetch_bullpen_team_stats(team)
            if stats:
                out[clean_key(team)] = stats
                logger.info("Loaded bullpen stats for %s", team)
        except Exception as exc:
            logger.warning("Failed bullpen scrape for %s: %s", team, exc)

    return out


def compute_bullpen_fatigue_score(team_stats):
    return round(
        (team_stats.get("last3_pitch_count", 0) / 100.0) +
        (team_stats.get("back_to_back_arms", 0) * 1.0) +
        (team_stats.get("three_in_four_arms", 0) * 1.5) +
        (team_stats.get("heavy_yesterday_arms", 0) * 1.0),
        2,
    )


def build_bullpen_risk_map(team_names=None):
    bullpen_daily_stats = fetch_bullpen_daily_stats(team_names) or {}
    bullpen_risk_map = {}
    bullpen_fatigue_scores = {}

    for team, stats in bullpen_daily_stats.items():
        score = compute_bullpen_fatigue_score(stats)
        bullpen_fatigue_scores[team] = score
        bullpen_risk_map[team] = fatigue_to_risk(score)

    return bullpen_risk_map, bullpen_fatigue_scores, bullpen_daily_stats


def normalize_pitcher_name(name):
    name = str(name).strip()
    if "," in name:
        last, first = [p.strip() for p in name.split(",", 1)]
        name = f"{first} {last}"
    return clean_key(name)


def load_batter_whiff_map(csv_path):
    df = pd.read_csv(csv_path, usecols=[0, 23])
    df.columns = ["name", "whiff_pct"]

    out = {}
    for _, row in df.iterrows():
        name = str(row["name"]).strip()
        if "," in name:
            last, first = [x.strip() for x in name.split(",", 1)]
            name = f"{first} {last}"

        try:
            out[clean_key(name)] = float(row["whiff_pct"])
        except Exception:
            continue

    return out


def get_team_drs_map():
    return {
        "arizona diamondbacks": -4,
        "athletics": -9,
        "atlanta braves": 12,
        "baltimore orioles": 3,
        "boston red sox": 1,
        "chicago cubs": 18,
        "chicago white sox": -6,
        "cincinnati reds": -2,
        "cleveland guardians": 7,
        "colorado rockies": -11,
        "detroit tigers": 6,
        "houston astros": 4,
        "kansas city royals": -3,
        "los angeles angels": -8,
        "los angeles dodgers": 14,
        "miami marlins": -5,
        "milwaukee brewers": 9,
        "minnesota twins": -1,
        "new york mets": 2,
        "new york yankees": 11,
        "philadelphia phillies": -7,
        "pittsburgh pirates": 5,
        "san diego padres": 8,
        "san francisco giants": 10,
        "seattle mariners": 13,
        "st louis cardinals": -4,
        "tampa bay rays": 6,
        "texas rangers": 15,
        "toronto blue jays": 7,
        "washington nationals": -6,
    }


def fetch_bullpen_team_stats(team_name):
    code = TEAM_CODE_MAP.get(clean_key(team_name))
    if not code:
        logger.warning("No team code for %s", team_name)
        return None

    site_code = INSIDETHEPEN_CODE_OVERRIDES.get(code, code)
    url = f"https://insidethepen.com/team/{site_code}-bullpen.html"
    response = requests.get(url, headers=UA, timeout=30)
    response.raise_for_status()

    tables = pd.read_html(StringIO(response.text))
    if not tables:
        logger.warning("No tables found for %s", team_name)
        return None

    usage_table = None
    for tbl in tables:
        cols = [str(c) for c in tbl.columns]
        if "Player" in cols and "NP-S" in cols:
            usage_table = tbl.copy()
            break

    if usage_table is None:
        logger.warning("No bullpen usage table found for %s", team_name)
        return None

    date_cols = [
        c for c in usage_table.columns
        if str(c) not in {"Player", "IP", "NP-S", "ERA"}
    ]

    back_to_back_arms = 0
    three_in_four_arms = 0
    heavy_yesterday_arms = 0
    last3_pitch_count = 0

    recent_cols = date_cols[:4]

    for _, row in usage_table.iterrows():
        appearances = 0
        last3_pitches = 0
        pitched_yesterday = False
        yesterday_pitches = 0

        for idx, col in enumerate(recent_cols):
            pitches, appeared = parse_pitch_count(row.get(col))
            if appeared:
                appearances += 1
                if idx < 3:
                    last3_pitches += pitches
                if idx == 0:
                    pitched_yesterday = True
                    yesterday_pitches = pitches

        if pitched_yesterday and yesterday_pitches >= 20:
            heavy_yesterday_arms += 1
        if appearances >= 2 and pitched_yesterday:
            back_to_back_arms += 1
        if appearances >= 3:
            three_in_four_arms += 1

        last3_pitch_count += last3_pitches

    return {
        "team": clean_key(team_name),
        "last3_pitch_count": last3_pitch_count,
        "back_to_back_arms": back_to_back_arms,
        "three_in_four_arms": three_in_four_arms,
        "heavy_yesterday_arms": heavy_yesterday_arms,
    }


def get_bullpen_risk_map():
    return {
        "new york yankees": 0,
        "toronto blue jays": 1,
        "st louis cardinals": 1,
        "los angeles dodgers": 0,
    }


def get_json(url, params=None):
    r = requests.get(url, params=params, timeout=30, headers=UA)
    r.raise_for_status()
    return r.json()


def schedule_for_date(date_str):
    data = get_json(f"{BASE}/schedule", params={"sportId": 1, "date": date_str, "hydrate": "probablePitcher,team"})
    return [g for d in data.get("dates", []) for g in d.get("games", [])]


def game_boxscore(game_pk):
    return get_json(f"{BASE}/game/{game_pk}/boxscore")


def player_info(player_id):
    data = get_json(f"{BASE}/people/{player_id}", params={"hydrate": "stats(group=[hitting,pitching],type=[season])"})
    return data["people"][0]


def player_split_stats(player_id, group, season, sit_codes=("vl", "vr", "h", "a")):
    try:
        codes_param = ",".join(sit_codes)
        data = get_json(
            f"{BASE}/people/{player_id}",
            params={"hydrate": f"stats(group=[{group}],type=[statSplits],sitCodes=[{codes_param}],season={season})"}
        )
        person = data["people"][0]
        out = {code: {} for code in sit_codes}
        for block in person.get("stats", []):
            for split in block.get("splits", []):
                code = split.get("code") or split.get("split", {}).get("code")
                if code in out:
                    out[code] = split.get("stat", {})
        return out
    except Exception:
        return {code: {} for code in sit_codes}


def stat_float(person, field):
    for block in person.get("stats", []):
        for split in block.get("splits", []):
            stat = split.get("stat", {})
            if stat.get(field) is not None:
                try:
                    return float(stat[field])
                except Exception:
                    return None
    return None


def recent_game_form(player_id, group, season):
    try:
        data = get_json(
            f"{BASE}/people/{player_id}",
            params={"hydrate": f"stats(group=[{group}],type=[gameLog],season={season})"}
        )
        person = data["people"][0]
        splits = []
        for block in person.get("stats", []):
            splits.extend(block.get("splits", []))
        splits = splits[-15:]
        if group == "hitting":
            vals = [s.get("stat", {}).get("avg") for s in splits if s.get("stat", {}).get("avg") is not None]
            return {"last15_avg": sum(map(float, vals)) / len(vals) if vals else None}
        vals = [s.get("stat", {}).get("era") for s in splits if s.get("stat", {}).get("era") is not None]
        return {"last15_era": sum(map(float, vals)) / len(vals) if vals else None}
    except Exception:
        return {}


def weather_for_game(game):
    try:
        venue = game.get("venue", {})
        lat = venue.get("location", {}).get("defaultCoordinates", {}).get("latitude")
        lon = venue.get("location", {}).get("defaultCoordinates", {}).get("longitude")
        if lat is None or lon is None:
            return {}
        data = requests.get(
            WEATHER,
            params={
                "latitude": lat,
                "longitude": lon,
                "current": "temperature_2m,wind_speed_10m,precipitation",
                "timezone": "auto",
            },
            timeout=20,
        ).json()
        cur = data.get("current", {})
        return {
            "temp_c": cur.get("temperature_2m"),
            "wind_kph": cur.get("wind_speed_10m"),
            "precip": cur.get("precipitation"),
        }
    except Exception:
        return {}


def mlb_hand(code):
    return "L" if str(code).upper().startswith("L") else "R"


def extract_lineup(box, side):
    team = box["teams"][side]
    rows = []
    for pid in team.get("batters", []):
        pdata = team["players"].get(f"ID{pid}", {})
        order = pdata.get("battingOrder")
        if not order:
            continue
        order_num = int(order) // 100
        if 1 <= order_num <= 9:
            rows.append({
                "player_id": pid,
                "name": pdata["person"]["fullName"],
                "order": order_num,
                "hand": mlb_hand(pdata.get("batSide", {}).get("code", "R")),
            })
    return sorted(rows, key=lambda x: x["order"])


def season_hit_rate(person):
    for block in person.get("stats", []):
        for split in block.get("splits", []):
            avg = split.get("stat", {}).get("avg")
            if avg is not None:
                try:
                    return float(avg)
                except Exception:
                    return None
    return None


def season_walk_rate(person):
    """BB / PA for the season. The MLB API doesn't expose walk rate as a
    direct field (unlike avg/obp/slg), so compute it from the raw counts."""
    for block in person.get("stats", []):
        for split in block.get("splits", []):
            stat = split.get("stat", {})
            bb = stat.get("baseOnBalls")
            pa = stat.get("plateAppearances")
            if bb is not None and pa:
                try:
                    return float(bb) / float(pa)
                except (TypeError, ZeroDivisionError):
                    return None
    return None


def choose_random_batter_df(df, min_score=None, top_only=False, seed=None):
    if df is None or df.empty:
        return pd.DataFrame()

    pool = df.copy()

    if min_score is not None and "hit_score" in pool.columns:
        pool = pool[pool["hit_score"] >= min_score].copy()

    if top_only and "game_rank" in pool.columns:
        pool = pool[pool["game_rank"] == 1].copy()

    if pool.empty:
        return pd.DataFrame()

    if seed is not None:
        return pool.sample(n=1, random_state=seed).reset_index(drop=True)

    return pool.sample(n=1).reset_index(drop=True)


_HITTING_FORM_CACHE_PATH = OUTPUT_DIR / "hitting_form_cache.json"
_hitting_form_cache = None  # lazy-loaded, shared across all calls in this process
_hitting_form_cache_lock = threading.Lock()


def _load_hitting_form_cache():
    global _hitting_form_cache
    if _hitting_form_cache is not None:
        return _hitting_form_cache
    try:
        with open(_HITTING_FORM_CACHE_PATH, "r") as f:
            _hitting_form_cache = json.load(f)
    except Exception:
        _hitting_form_cache = {}
    return _hitting_form_cache


def _save_hitting_form_cache():
    if _hitting_form_cache is None:
        return
    try:
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        with open(_HITTING_FORM_CACHE_PATH, "w") as f:
            json.dump(_hitting_form_cache, f)
    except Exception:
        logger.warning("Could not write hitting form cache to %s", _HITTING_FORM_CACHE_PATH)


def player_recent_hitting_form(player_id, season, as_of_date=None):
    """Single game-log fetch that derives everything we need from recent
    hitting performance: rolling averages, previous-game result, and the
    current consecutive-games-with-a-hit streak. Replaces what used to be
    two separate API calls (recent_game_form + previous_game_hit_info).

    Cached per (player_id, as_of_date) on disk -- the MLB gameLog endpoint
    always returns the full season with no way to fetch just new games, so
    the practical win is skipping repeat fetches within the same slate date
    (e.g. running lineups.sh twice in one day)."""
    cache_key = f"{player_id}:{as_of_date}" if as_of_date else None
    cache = _load_hitting_form_cache() if cache_key else None
    if cache is not None and cache_key in cache:
        return cache[cache_key]

    empty = {
        "last15_avg": None,
        "last5_avg": None,
        "prev_game_hits": None,
        "prev_game_zero_hit_flag": 0,
        "current_hit_streak": 0,
        "hot_streak_flag": 0,
    }
    try:
        data = get_json(
            f"{BASE}/people/{player_id}",
            params={"hydrate": f"stats(group=[hitting],type=[gameLog],season={season})"},
        )
        person = data["people"][0]

        splits = []
        for block in person.get("stats", []):
            splits.extend(block.get("splits", []))

        if not splits:
            return empty

        # splits come back oldest -> newest; only count games the batter
        # actually appeared at the plate in (skip 0-AB games like a DNP
        # entry) so a day off doesn't wrongly reset the streak.
        played = [s for s in splits if safe_int_or_none((s.get("stat", {}) or {}).get("atBats")) not in (None, 0)]

        last15 = played[-15:]
        last5 = played[-5:]

        def _avg_of(games):
            vals = [g.get("stat", {}).get("avg") for g in games if g.get("stat", {}).get("avg") is not None]
            return sum(map(float, vals)) / len(vals) if vals else None

        last_game = played[-1] if played else None
        prev_hits = safe_int_or_none((last_game.get("stat", {}) or {}).get("hits")) if last_game else None

        streak = 0
        for g in reversed(played):
            hits = safe_int_or_none((g.get("stat", {}) or {}).get("hits"))
            if hits is None:
                break
            if hits >= 1:
                streak += 1
            else:
                break

        result = {
            "last15_avg": _avg_of(last15),
            "last5_avg": _avg_of(last5),
            "prev_game_hits": prev_hits,
            "prev_game_zero_hit_flag": int(prev_hits == 0) if prev_hits is not None else 0,
            "current_hit_streak": streak,
            "hot_streak_flag": int(streak >= HOT_STREAK_MIN_GAMES),
        }
        if cache_key:
            with _hitting_form_cache_lock:
                _load_hitting_form_cache()[cache_key] = result
        return result
    except Exception:
        return empty


def build_rankings(date_str, max_workers=8, top_n_per_game=1, min_hit_streak=0, hottest_only=False):
    year = int(date_str[:4])
    results = []
    cache = {}
    todays_teams = set()

    tasks = []

    for g in schedule_for_date(date_str):
        game_pk = g["gamePk"]
        away = g["teams"]["away"]["team"]["name"]
        home = g["teams"]["home"]["team"]["name"]
        todays_teams.add(clean_key(away))
        todays_teams.add(clean_key(home))

        matchup = f"{away} @ {home}"
        box = game_boxscore(game_pk)
        weather = weather_for_game(g)

        for bat_side, opp_side in [("away", "home"), ("home", "away")]:
            lineup = extract_lineup(box, bat_side)
            if not lineup:
                continue

            pitchers = box["teams"][opp_side].get("pitchers", [])
            if not pitchers:
                continue

            starter_id = pitchers[0]
            cache.setdefault(starter_id, player_info(starter_id))
            sp = cache[starter_id]
            sp_name = sp["fullName"]
            sp_hand = mlb_hand(sp.get("pitchHand", {}).get("code", "R"))
            sp_era = stat_float(sp, "era")
            sp_whip = stat_float(sp, "whip")
            sp_splits = player_split_stats(starter_id, "pitching", year)

            batting_team = away if bat_side == "away" else home
            opp_team = home if bat_side == "away" else away

            for batter in [b for b in lineup if b["order"] <= 9]:
                tasks.append({
                    "game_pk": game_pk,
                    "matchup": matchup,
                    "batting_team": batting_team,
                    "opp_team": opp_team,
                    "bat_side": bat_side,
                    "batter": batter,
                    "starter_id": starter_id,
                    "sp": sp,
                    "sp_name": sp_name,
                    "sp_hand": sp_hand,
                    "sp_era": sp_era,
                    "sp_whip": sp_whip,
                    "sp_splits": sp_splits,
                    "weather": weather,
                    "year": year,
                })

    def fetch_batter_row(task):
        batter = task["batter"]
        starter_id = task["starter_id"]
        sp = task["sp"]
        sp_name = task["sp_name"]
        sp_hand = task["sp_hand"]
        sp_era = task["sp_era"]
        sp_whip = task["sp_whip"]
        sp_splits = task["sp_splits"]
        weather = task["weather"]
        game_pk = task["game_pk"]
        year = task["year"]

        bp = player_info(batter["player_id"])

        bat_splits_all = player_split_stats(batter["player_id"], "hitting", year)
        bat_split = bat_splits_all.get("vr" if sp_hand == "R" else "vl", {})
        home_away_split = bat_splits_all.get("h" if task["bat_side"] == "home" else "a", {})
        pit_split = sp_splits.get("vr" if batter["hand"] == "R" else "vl", {})

        hitting_form = player_recent_hitting_form(batter["player_id"], year, as_of_date=date_str)

        return {
            "date": date_str,
            "game_pk": game_pk,
            "matchup": task["matchup"],
            "batting_team": task["batting_team"],
            "opp_team": task["opp_team"],
            "batter": batter["name"],
            "batter_id": batter["player_id"],
            "order": batter["order"],
            "batter_hand": batter["hand"],
            "opp_pitcher": sp_name,
            "opp_pitcher_hand": sp_hand,
            "batter_avg": season_hit_rate(bp),
            "batter_obp": stat_float(bp, "obp"),
            "batter_slg": stat_float(bp, "slg"),
            "batter_bb_rate": season_walk_rate(bp),
            "batter_vs_hand_avg": bat_split.get("avg"),
            "batter_vs_hand_obp": bat_split.get("obp"),
            "batter_vs_hand_slg": bat_split.get("slg"),
            "batter_home_away_avg": home_away_split.get("avg"),
            "batter_home_away_obp": home_away_split.get("obp"),
            "batter_home_away_slg": home_away_split.get("slg"),
            "batting_home_flag": 1 if task["bat_side"] == "home" else 0,
            "pitcher_era": sp_era,
            "pitcher_whip": sp_whip,
            "pitcher_ba_allowed": stat_float(sp, "avg"),
            "pitcher_xba_allowed": stat_float(sp, "xBA"),
            "pitcher_woba_allowed": stat_float(sp, "wOBA"),
            "pitcher_vs_hand_baa": pit_split.get("avg"),
            "pitcher_vs_hand_xba": pit_split.get("xba"),
            "pitcher_vs_hand_woba": pit_split.get("woba"),
            "last15_hitter_avg": hitting_form.get("last15_avg"),
            "last15_pitcher_era": recent_game_form(starter_id, "pitching", year).get("last15_era"),
            "last5_hitter_avg": hitting_form.get("last5_avg"),
            "prev_game_hits": hitting_form.get("prev_game_hits"),
            "prev_game_zero_hit_flag": hitting_form.get("prev_game_zero_hit_flag"),
            "current_hit_streak": hitting_form.get("current_hit_streak"),
            "hot_streak_flag": hitting_form.get("hot_streak_flag"),
            "temp_c": weather.get("temp_c"),
            "wind_kph": weather.get("wind_kph"),
            "precip": weather.get("precip"),
        }

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(fetch_batter_row, t): t for t in tasks}
        for fut in as_completed(futures):
            try:
                row = fut.result()
                results.append(row)
            except Exception as exc:
                logger.warning("Batter row failed: %s", exc)

    _save_hitting_form_cache()

    BASE_DIR = Path(__file__).resolve().parent
    batter_whiff_map = load_batter_whiff_map(BASE_DIR / "batter_whiff.csv")
    team_drs_map = get_team_drs_map()
    bullpen_risk_map, bullpen_fatigue_scores, bullpen_daily_stats = build_bullpen_risk_map(todays_teams)

    batter_whiff_map = {clean_key(k): v for k, v in batter_whiff_map.items()}
    team_drs_map = {clean_key(k): v for k, v in team_drs_map.items()}
    bullpen_risk_map = {clean_key(k): v for k, v in bullpen_risk_map.items()}

    logger.info("bullpen teams found: %s", len(bullpen_risk_map))
    for team in sorted(bullpen_risk_map)[:10]:
        logger.debug(
            "%s stats=%s score=%s risk=%s",
            team,
            bullpen_daily_stats.get(team),
            bullpen_fatigue_scores.get(team),
            bullpen_risk_map.get(team),
        )

    df = pd.DataFrame(results)
    if df.empty:
        return df

    df["batter_whiff_pct"] = pd.to_numeric(
        df["batter"].map(lambda x: batter_whiff_map.get(clean_key(x))),
        errors="coerce",
    ).fillna(25.0)

    df["opp_team_defense_drs"] = pd.to_numeric(
        df["opp_team"].map(lambda x: team_drs_map.get(clean_key(x))),
        errors="coerce",
    ).fillna(0)

    df["bullpen_risk_flag"] = pd.to_numeric(
        df["opp_team"].map(lambda x: bullpen_risk_map.get(clean_key(x), 0)),
        errors="coerce",
    ).fillna(0)

    df["bullpen_fatigue_score"] = pd.to_numeric(
        df["opp_team"].map(lambda x: bullpen_fatigue_scores.get(clean_key(x), 0)),
        errors="coerce",
    ).fillna(0)

    df["bullpen_adjustment"] = df["bullpen_risk_flag"] * 0.02

    logger.debug("df columns before context features: %s", df.columns.tolist())
    logger.debug("df preview:\n%s", df.head(3).to_string())

    if "pitcher_whiff_pct" not in df.columns:
        df["pitcher_whiff_pct"] = 0.0

    df["high_whiff_flag"] = (df["pitcher_whiff_pct"] >= 25).astype(int)
    df["elite_defense_flag"] = (df["opp_team_defense_drs"] >= 8).astype(int)
    df["weak_defense_flag"] = (df["opp_team_defense_drs"] <= -5).astype(int)

    df["context_adjustment"] = (
        df["high_whiff_flag"] * -0.04 +
        df["elite_defense_flag"] * -0.03 +
        df["weak_defense_flag"] * 0.02 +
        df["bullpen_adjustment"]
    )

    df["whiff_penalty"] = 0.0
    df.loc[df["batter_whiff_pct"] >= 35, "whiff_penalty"] = 0.05
    df.loc[(df["batter_whiff_pct"] >= 30) & (df["batter_whiff_pct"] < 35), "whiff_penalty"] = 0.03
    df.loc[(df["batter_whiff_pct"] >= 25) & (df["batter_whiff_pct"] < 30), "whiff_penalty"] = 0.01

    df["order_score"] = 1 - ((safe_num(df["order"], default=5) - 1) / 4)
    df["batter_avg_score"] = safe_norm(df["batter_avg"], default=0.5)

    df["batter_split_score"] = (
        safe_norm(df["batter_vs_hand_avg"], default=0.5) * 0.45 +
        safe_norm(df["batter_vs_hand_obp"], default=0.5) * 0.25 +
        safe_norm(df["batter_vs_hand_slg"], default=0.5) * 0.30
    )

    df["pitcher_score"] = (
        safe_norm(df["pitcher_era"], default=0.5) * 0.20 +
        safe_norm(df["pitcher_whip"], default=0.5) * 0.15 +
        safe_norm(df["pitcher_ba_allowed"], default=0.5) * 0.15 +
        safe_norm(df["pitcher_xba_allowed"], default=0.5) * 0.15 +
        safe_norm(df["pitcher_woba_allowed"], default=0.5) * 0.10 +
        safe_norm(df["pitcher_vs_hand_baa"], default=0.5) * 0.10 +
        safe_norm(df["pitcher_vs_hand_xba"], default=0.5) * 0.075 +
        safe_norm(df["pitcher_vs_hand_woba"], default=0.5) * 0.075
    )

    df["form_score"] = (
        safe_norm(df["last15_hitter_avg"], default=0.5) * 0.7 +
        safe_norm(df["last15_pitcher_era"], default=0.5) * 0.3
    )

    df["env_score"] = (
        safe_norm(df["temp_c"], default=0.5) * 0.4 +
        safe_norm(df["wind_kph"], default=0.5) * 0.3 +
        (1 - safe_norm(df["precip"], default=0.0)).clip(0, 1) * 0.3
    )

    same_hand_penalty = (df["batter_hand"] == df["opp_pitcher_hand"]).astype(int) * 0.05

    df["current_hit_streak"] = pd.to_numeric(df["current_hit_streak"], errors="coerce").fillna(0)
    df["hot_streak_flag"] = pd.to_numeric(df["hot_streak_flag"], errors="coerce").fillna(0).astype(int)

    # Relative ranking across today's ENTIRE slate (not just the per-game
    # winners) -- this is what actually identifies who's hottest/coldest
    # today, rather than everyone above a fixed absolute cutoff.
    df["streak_rank_today"] = df["current_hit_streak"].rank(method="min", ascending=False).astype(int)
    df["streak_percentile_today"] = df["current_hit_streak"].rank(pct=True)
    top_n = max(1, round(len(df) * 0.10))  # top ~10% of today's full player pool
    df["hottest_today_flag"] = (df["streak_rank_today"] <= top_n).astype(int)
    df["coldest_today_flag"] = (
        (df["current_hit_streak"] == 0) & (df["prev_game_zero_hit_flag"] == 1)
    ).astype(int)

    # small, capped bonus -- a longer streak helps a little, but this should
    # never dominate the pitcher/split/form signals that do the heavy lifting
    df["hot_streak_bonus"] = df["current_hit_streak"].clip(0, 10) * 0.01

    df["context_adjustment"] = (
        df["high_whiff_flag"] * -0.04 +
        df["elite_defense_flag"] * -0.03 +
        df["weak_defense_flag"] * 0.02 +
        df["bullpen_risk_flag"] * 0.03 +
        df["hot_streak_bonus"]
    )

    df["hit_score"] = (
        df["order_score"] * 0.20 +
        df["batter_split_score"] * 0.25 +
        df["pitcher_score"] * 0.30 +
        df["form_score"] * 0.15 +
        df["env_score"] * 0.10
        - same_hand_penalty
        - df["whiff_penalty"]
        + df["context_adjustment"]
    ).clip(0, 1)

    logger.info("rows built: %s", len(df))

    if min_hit_streak > 0:
        before = len(df)
        df = df[df["current_hit_streak"] >= min_hit_streak].copy()
        logger.info(
            "Filtered to batters with a %s+ game hit streak: %s -> %s rows",
            min_hit_streak, before, len(df)
        )
        if df.empty:
            return df

    if hottest_only:
        before = len(df)
        df = df[df["hottest_today_flag"] == 1].copy()
        logger.info(
            "Filtered to hottest ~10%% of today's slate by streak: %s -> %s rows",
            before, len(df)
        )
        if df.empty:
            return df

    df["game_rank"] = df.groupby("matchup")["hit_score"].rank(
        method="first",
        ascending=False,
    ).astype(int)

    df = df[df["game_rank"] <= top_n_per_game].copy()

    return df.sort_values(["matchup", "game_rank", "hit_score"], ascending=[True, True, False])


def apply_sigmoid_calibration(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty or "hit_score" not in df.columns:
        return df

    X = df[["hit_score"]].values
    y = (df["had_hit"] > 0).astype(int).values if "had_hit" in df.columns else (df["game_rank"] == 1).astype(int).values

    base_clf = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=1000, solver="lbfgs")),
        ]
    )

    cal_clf = CalibratedClassifierCV(base_clf, method="sigmoid", cv=3)
    cal_clf.fit(X, y)

    probs = cal_clf.predict_proba(X)[:, 1]
    df = df.copy()
    df["hit_prob_cal"] = probs
    return df

def calibrate_hit_probs(df: pd.DataFrame, method: str = "sigmoid") -> pd.DataFrame:
    """
    Fit two small calibration models on top of hit_score:
      - prob_hit_1: P(hit >= 1)
      - prob_hit_2: P(hit >= 2)

    For now, targets are approximated from game_rank and a simple rule.
    When you have a graded_history with real hit_count, you can replace
    these proxy labels with real ones.
    """
    if df.empty or "hit_score" not in df.columns:
        return df

    df = df.copy()

    # Proxy targets for now:
    # - Assume top-ranked batter per matchup is more likely to get a hit.
    # - Assume top 1 per matchup is more likely to multi-hit than lower ranks.
    if "had_hit" in df.columns:
        y1 = df["had_hit"].astype(int).values
    else:
        # Rough proxy: game_rank == 1 => more likely to get a hit
        y1 = (df["game_rank"] == 1).astype(int).values

    if "hit_count" in df.columns:
        y2 = (df["hit_count"] >= 2).astype(int).values
    else:
        # Rough proxy: only top batter per matchup can multi-hit
        y2 = (df["game_rank"] == 1).astype(int).values

    X = df[["hit_score"]].values

    base_clf = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=1000, solver="lbfgs")),
        ]
    )

    # P(hit >= 1)
    cal1 = CalibratedClassifierCV(base_clf, method=method, cv=3)
    cal1.fit(X, y1)
    df["prob_hit_1"] = cal1.predict_proba(X)[:, 1]

    # P(hit >= 2)
    cal2 = CalibratedClassifierCV(base_clf, method=method, cv=3)
    cal2.fit(X, y2)
    df["prob_hit_2"] = cal2.predict_proba(X)[:, 1]

    return df


def load_hit_models():
    """
    Load the trained models:
      - had_hit_1: P(hit >= 1)
      - had_hit_2: P(hit >= 2)
      - had_run_1: P(run scored >= 1)

    Expects bundles in output/:
      - had_hit_1_model_bundle.pkl
      - had_hit_2_model_bundle.pkl
      - had_run_1_model_bundle.pkl (optional; older models may not have this yet)
    """
    base = Path(__file__).resolve().parent
    output_dir = base.parent / "output"

    bundle_1 = output_dir / "had_hit_1_model_bundle.pkl"
    bundle_2 = output_dir / "had_hit_2_model_bundle.pkl"
    bundle_3 = output_dir / "had_run_1_model_bundle.pkl"

    if not bundle_1.exists() or not bundle_2.exists():
        logger.warning(
            "One or both hit model bundles not found. "
            "Run train_hit_model.py first to generate had_hit_1 and had_hit_2 models."
        )
        return None, None, None

    with open(bundle_1, "rb") as f:
        model_1 = pickle.load(f)

    with open(bundle_2, "rb") as f:
        model_2 = pickle.load(f)

    model_3 = None
    if bundle_3.exists():
        with open(bundle_3, "rb") as f:
            model_3 = pickle.load(f)
    else:
        logger.info(
            "had_run_1_model_bundle.pkl not found; run probability will be skipped "
            "until you retrain with the updated train_hit_model.py."
        )

    return model_1, model_2, model_3


def score_with_hit_models(df: pd.DataFrame, model_1, model_2, model_3=None) -> pd.DataFrame:
    """
    Score a DataFrame of today's candidates with the trained models.
    Adds:
      - prob_hit_1: P(hit >= 1)
      - prob_hit_2: P(hit >= 2)
      - prob_run_1: P(run scored >= 1), if a run model is available
    """
    if df.empty or model_1 is None or model_2 is None:
        return df

    df = df.copy()

    def _score(model, out_col):
        feature_cols = model["feature_columns"]
        missing = [c for c in feature_cols if c not in df.columns]
        if missing:
            logger.warning(
                "Today's file missing some model features for %s: %s. "
                "Scoring may be degraded.",
                out_col,
                missing,
            )
        X = df[feature_cols].copy()
        pipeline: Pipeline = model["pipeline"]
        df[out_col] = pipeline.predict_proba(X)[:, 1]

    _score(model_1, "prob_hit_1")
    _score(model_2, "prob_hit_2")

    if model_3 is not None:
        _score(model_3, "prob_run_1")

    return df


def main():
    parser = argparse.ArgumentParser(
        description="Find top MLB batters in favorable opposite-handed matchups and rank hit likelihood."
    )

    parser.add_argument("--date", default=dt.date.today().isoformat())
    parser.add_argument("--top", type=int, default=5, help="rows to print per matchup (display only)")
    parser.add_argument(
        "--picks-per-game",
        type=int,
        default=1,
        help="how many ranked batters per game to keep in the output (default: 1, i.e. best hitter only)",
    )
    parser.add_argument(
        "--min-hit-streak",
        type=int,
        default=0,
        help=f"only include batters on at least this many consecutive games with a hit "
        f"(0 = no filter; a batter is flagged 'hot' at {HOT_STREAK_MIN_GAMES}+ games)",
    )
    parser.add_argument(
        "--hottest-only",
        action="store_true",
        help="only include batters in the hottest ~10%% of today's full slate by current hit streak "
        "(relative ranking, not a fixed threshold)",
    )
    parser.add_argument(
        "-v", "--verbose",
        action="count",
        default=0,
        help="increase terminal verbosity (-v for progress, -vv for debug)",
    )

    parser.add_argument(
        "--random-pick",
        action="store_true",
        help="write one random batter from the ranked pool",
    )

    parser.add_argument(
        "--random-min-score",
        type=float,
        default=None,
        help="minimum hit_score required for random pick",
    )

    parser.add_argument(
        "--random-top-only",
        action="store_true",
        help="random pick only from game_rank == 1 rows",
    )

    parser.add_argument(
        "--random-seed",
        type=int,
        default=None,
        help="optional seed for reproducible random pick",
    )

    parser.add_argument(
        "--workers",
        type=int,
        default=8,
        help="number of worker threads for API calls",
    )

    parser.add_argument(
        "--calibrate",
        action="store_true",
        help="apply sigmoid calibration to hit_score to get calibrated probabilities",
    )

    args = parser.parse_args()

    setup_logging(args.verbose)

    df = build_rankings(
        args.date,
        max_workers=args.workers,
        top_n_per_game=args.picks_per_game,
        min_hit_streak=args.min_hit_streak,
        hottest_only=args.hottest_only,
    )

    # Load and apply trained hit/run models if available
    model_1, model_2, model_3 = load_hit_models()
    if model_1 is not None and model_2 is not None:
        df = score_with_hit_models(df, model_1, model_2, model_3)

    if df.empty:
        print(f"No posted lineups or no matching batters for {args.date}")
        return

    if args.calibrate:
        df = calibrate_hit_probs(df, method="sigmoid")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_csv = OUTPUT_DIR / f"hit_candidates_{args.date}.csv"
    df.to_csv(out_csv, index=False)
    print(f"Wrote {out_csv}")

    if args.random_pick:
        random_pick_df = choose_random_batter_df(
            df,
            min_score=args.random_min_score,
            top_only=args.random_top_only,
            seed=args.random_seed,
        )

        if random_pick_df.empty:
            print("\nRandom batter pick: no batter matched the random filters.")
        else:
            random_csv = OUTPUT_DIR / f"random_batter_pick_{args.date}.csv"
            random_pick_df.to_csv(random_csv, index=False)

            rp = random_pick_df.iloc[0]
            print("\nRandom batter pick:")
            print(
                f"{rp.get('batter', 'Unknown')} | "
                f"{rp.get('matchup', 'Unknown matchup')} | "
                f"order={rp.get('order', 'NA')} | "
                f"score={rp.get('hit_score', 'NA')}"
            )

    zero_hit_side_list = pd.DataFrame()
    if "prev_game_zero_hit_flag" in df.columns:
        zero_hit_side_list = df[df["prev_game_zero_hit_flag"] == 1].copy()
        zero_hit_side_list = zero_hit_side_list.sort_values(
            by=["hit_score", "game_rank"],
            ascending=[False, True],
        )

        if not zero_hit_side_list.empty:
            print("\n0-hit previous game side list")
            cols = [
                "batter",
                "matchup",
                "order",
                "batter_hand",
                "opp_pitcher",
                "opp_pitcher_hand",
                "prev_game_hits",
                "hit_score",
                "game_rank",
            ]
            cols = [c for c in cols if c in zero_hit_side_list.columns]
            print(zero_hit_side_list[cols].head(10).to_string(index=False))

    for matchup, grp in df.groupby("matchup"):
        print(f"\n{matchup}")
        cols = [
            "game_rank", "batter", "order", "batter_hand",
            "opp_pitcher", "opp_pitcher_hand", "hit_score",
            "pitcher_whiff_pct", "high_whiff_flag",
            "opp_team_defense_drs", "elite_defense_flag",
            "bullpen_risk_flag",
            "pitcher_era", "pitcher_whip", "batter_avg", "batter_bb_rate",
            "batter_home_away_avg", "batting_home_flag",
            "current_hit_streak", "streak_rank_today", "hottest_today_flag",
        ]
        if "prob_hit_1" in grp.columns:
            cols.append("prob_hit_1")
        if "prob_hit_2" in grp.columns:
            cols.append("prob_hit_2")
        if "prob_run_1" in grp.columns:
            cols.append("prob_run_1")
        cols = [c for c in cols if c in grp.columns]
        print(grp[cols].head(args.top).to_string(index=False))


if __name__ == "__main__":
    main()