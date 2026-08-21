from __future__ import annotations

import datetime as dt

from django.shortcuts import render

# adjust this import to your real module path
from .services import build_rankings


def _to_float(value, default=None):
    try:
        if value in (None, ""):
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _to_int(value, default=None):
    try:
        if value in (None, ""):
            return default
        return int(value)
    except (TypeError, ValueError):
        return default


def _pick_featured_cards(rows):
    if not rows:
        return []

    sorted_rows = sorted(
        rows,
        key=lambda x: (
            -(x.get("hit_score") or 0),
            x.get("game_rank") or 999,
            x.get("order") or 999,
        )
    )

    recommended = sorted_rows[0]

    decoy = None
    for row in sorted_rows[1:]:
        if row.get("matchup") == recommended.get("matchup"):
            decoy = row
            break
    if decoy is None and len(sorted_rows) > 1:
        decoy = sorted_rows[1]

    upside = None
    for row in sorted_rows[1:]:
        if row != decoy:
            upside = row
            break

    featured = []
    if decoy:
        featured.append({
            **decoy,
            "card_role": "decoy",
            "card_title": "Steady alternative",
        })

    featured.append({
        **recommended,
        "card_role": "recommended",
        "card_title": "Best hit spot",
    })

    if upside:
        featured.append({
            **upside,
            "card_role": "upside",
            "card_title": "Higher variance",
        })

    return featured[:3]


def dashboard(request):
    slate_date = request.GET.get("date") or dt.date.today().isoformat()
    team_filter = (request.GET.get("team") or "").strip().lower()
    min_score = _to_float(request.GET.get("min_score"))
    show_zero_hit = request.GET.get("show_zero_hit") or ""
    sort = request.GET.get("sort") or "hit_score"

    df = build_rankings(slate_date)

    if df is None or df.empty:
        context = {
            "slate_date": slate_date,
            "featured_picks": [],
            "all_hitters": [],
            "zero_hit_watch": [],
            "filters": {
                "date": slate_date,
                "team": team_filter,
                "min_score": request.GET.get("min_score", ""),
                "show_zero_hit": show_zero_hit,
                "sort": sort,
            },
        }
        return render(request, "mlb/dashboard.html", context)

    if team_filter:
        if "batting_team" in df.columns:
            df = df[df["batting_team"].fillna("").str.lower().str.contains(team_filter)]
        elif "matchup" in df.columns:
            df = df[df["matchup"].fillna("").str.lower().str.contains(team_filter)]

    if min_score is not None and "hit_score" in df.columns:
        df = df[df["hit_score"] >= min_score].copy()

    if show_zero_hit == "1" and "prev_game_zero_hit_flag" in df.columns:
        df = df[df["prev_game_zero_hit_flag"] == 1].copy()

    if sort == "order" and "order" in df.columns:
        df = df.sort_values(by=["order", "hit_score"], ascending=[True, False])
    elif sort == "pitcher_era" and "pitcher_era" in df.columns:
        df = df.sort_values(by=["pitcher_era", "hit_score"], ascending=[False, False])
    else:
        if "hit_score" in df.columns and "game_rank" in df.columns:
            df = df.sort_values(by=["hit_score", "game_rank"], ascending=[False, True])

    all_hitters = df.to_dict(orient="records")

    zero_hit_watch = []
    if "prev_game_zero_hit_flag" in df.columns:
        zero_hit_watch = [
            row for row in all_hitters
            if row.get("prev_game_zero_hit_flag") == 1
        ][:10]

    featured_picks = _pick_featured_cards(all_hitters)

    context = {
        "slate_date": slate_date,
        "featured_picks": featured_picks,
        "all_hitters": all_hitters,
        "zero_hit_watch": zero_hit_watch,
        "filters": {
            "date": slate_date,
            "team": request.GET.get("team", ""),
            "min_score": request.GET.get("min_score", ""),
            "show_zero_hit": show_zero_hit,
            "sort": sort,
        },
    }
    return render(request, "mlb/dashboard.html", context)