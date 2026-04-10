"""
PropEdge V2.0 — batch_grade.py
B0 job: Runs after games complete (07:00 UK).

Actions:
  1. Fetch yesterday's box scores via NBA API (ScoreboardV3 + BoxScoreTraditionalV3)
  2. Grade all open plays in today.json + season JSONs
  3. Append game rows to gamelogs_2025_26.csv → recompute rolling stats
  4. Write post-match narratives (lossType, postMatchReason, delta)
  5. Update monthly split files
  6. Rebuild H2H + DVP
  7. Git push

WIN  = model direction was correct (OVER→actual>line, UNDER→actual<=line)
LOSS = direction wrong
DNP  = player not in box score
PUSH = actual == line exactly
"""
from __future__ import annotations

import json
import re
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).parent.resolve()
sys.path.insert(0, str(ROOT))

from config import (
    VERSION_TAG, FILE_TODAY, FILE_SEASON_2425, FILE_SEASON_2526,
    FILE_GL_2526, FILE_GL_2425, FILE_H2H, FILE_DVP,
    clean_json, now_uk,
)
from player_name_aliases import _norm, resolve_grade_name
from reasoning_engine import generate_post_match_reason
from monthly_split import update_month
from rolling_engine import recompute_rolling


# ── Helpers ───────────────────────────────────────────────────────────────────

def _si(v):
    try: return int(v) if pd.notna(v) else 0
    except: return 0

def _parse_min(v) -> float:
    s = str(v).strip()
    if s in ("", "None", "nan", "0", "PT00M00.00S"): return 0.0
    if s.startswith("PT") and "M" in s:
        m = re.match(r"PT(\d+)M([\d.]+)S", s)
        return float(m.group(1)) + float(m.group(2)) / 60 if m else 0.0
    if ":" in s:
        p = s.split(":"); return float(p[0]) + float(p[1]) / 60
    try: return float(s)
    except: return 0.0


# ── Step 1: Fetch box scores ──────────────────────────────────────────────────

def fetch_box_scores(date_str: str) -> tuple[list[dict], dict[str, float], set[str]]:
    """
    Fetch box scores using nba_api V3 endpoints.
    Returns (played_rows, results_map, players_in_box).
    """
    print(f"\n  Fetching box scores: {date_str}...")

    played_rows:    list[dict] = []
    players_in_box: set[str]  = set()
    results_map:    dict[str, float] = {}

    # Bio cache from existing game log
    bio: dict[int, dict] = {}
    if FILE_GL_2526.exists():
        try:
            df26 = pd.read_csv(FILE_GL_2526, low_memory=False)
            bc = ["PLAYER_ID", "PLAYER_NAME", "PLAYER_POSITION", "PLAYER_POSITION_FULL",
                  "PLAYER_CURRENT_TEAM", "GAME_TEAM_ABBREVIATION", "GAME_TEAM_NAME"]
            bc_avail = [c for c in bc if c in df26.columns]
            for _, r in df26.drop_duplicates("PLAYER_ID", keep="last")[bc_avail].iterrows():
                try: bio[int(r["PLAYER_ID"])] = r.to_dict()
                except: pass
        except Exception as e:
            print(f"  ⚠ Bio cache: {e}")

    # Step 1a: ScoreboardV3 → game IDs
    game_ids: list[str] = []
    ctx: dict[str, dict] = {}
    try:
        from nba_api.stats.endpoints import ScoreboardV3
        for attempt in range(3):
            try:
                time.sleep(1 + attempt * 2)
                sb = ScoreboardV3(game_date=date_str, league_id="00")
                gh = sb.game_header.get_data_frame()
                ls = sb.line_score.get_data_frame()
                if gh.empty:
                    print("  No games found for date.")
                    return played_rows, results_map, players_in_box
                game_ids = gh["gameId"].tolist()
                print(f"  ScoreboardV3: {len(game_ids)} games")
                for g in game_ids:
                    r = ls[ls["gameId"] == g]
                    if len(r) >= 2:
                        ctx[str(g)] = {
                            "htid": r.iloc[0]["teamId"],
                            "ht":   r.iloc[0]["teamTricode"],
                            "at":   r.iloc[1]["teamTricode"],
                            "hs":   _si(r.iloc[0].get("score", 0)),
                            "as_":  _si(r.iloc[1].get("score", 0)),
                        }
                break
            except Exception as e:
                print(f"  ⚠ ScoreboardV3 attempt {attempt+1}/3: {e}")
                if attempt == 2: raise
    except Exception as e:
        print(f"  ⚠ ScoreboardV3 failed: {e}")

    if not game_ids:
        print("  ⚠ No game IDs — falling back to CSV lookup")
        return _csv_fallback(date_str, played_rows, results_map, players_in_box)

    # Step 1b: BoxScoreTraditionalV3 per game
    try:
        from nba_api.stats.endpoints import BoxScoreTraditionalV3
    except ImportError:
        print("  ⚠ BoxScoreTraditionalV3 not available — check nba_api version")
        return _csv_fallback(date_str, played_rows, results_map, players_in_box)

    for g in game_ids:
        ps = None
        for attempt in range(3):
            try:
                time.sleep(0.8 + attempt * 2)
                box = BoxScoreTraditionalV3(game_id=g)
                ps  = box.player_stats.get_data_frame()
                break
            except Exception as e:
                print(f"  ⚠ BoxScore game {g} attempt {attempt+1}/3: {e}")
                if attempt == 2:
                    print(f"  ✗ Skipping game {g}")
        if ps is None or ps.empty:
            continue

        try:
            col_map = {
                "personId": "PLAYER_ID", "teamId": "TEAM_ID",
                "teamTricode": "TEAM_ABBREVIATION",
                "firstName": "FN", "familyName": "LN", "minutes": "MR",
                "fieldGoalsMade": "FGM", "fieldGoalsAttempted": "FGA",
                "threePointersMade": "FG3M", "threePointersAttempted": "FG3A",
                "freeThrowsMade": "FTM", "freeThrowsAttempted": "FTA",
                "reboundsOffensive": "OREB", "reboundsDefensive": "DREB",
                "reboundsTotal": "REB", "assists": "AST", "steals": "STL",
                "blocks": "BLK", "turnovers": "TOV", "foulsPersonal": "PF",
                "points": "PTS", "plusMinusPoints": "PLUS_MINUS",
            }
            ps = ps.rename(columns={k: v for k, v in col_map.items() if k in ps.columns})
            if "PLAYER_NAME" not in ps.columns and "FN" in ps.columns:
                ps["PLAYER_NAME"] = (
                    ps["FN"].fillna("").str.strip() + " " +
                    ps["LN"].fillna("").str.strip()
                ).str.strip()

            c = ctx.get(str(g), {})

            for _, p in ps.iterrows():
                pname = str(p.get("PLAYER_NAME", "")).strip()
                if pname:
                    players_in_box.add(pname)

                mn = _parse_min(p.get("MR", 0))
                if mn <= 0:
                    continue

                pid = _si(p.get("PLAYER_ID", 0))
                tid = _si(p.get("TEAM_ID", 0))
                ta  = str(p.get("TEAM_ABBREVIATION", ""))
                ih  = 1 if tid == c.get("htid") else 0
                opp_rows = ps[ps["TEAM_ID"] != tid]["TEAM_ABBREVIATION"]
                opp      = opp_rows.iloc[0] if len(opp_rows) > 0 else "UNK"
                mu       = f"{ta} vs. {opp}" if ih else f"{ta} @ {opp}"
                wl       = ("W" if c.get("hs", 0) > c.get("as_", 0) else "L") if ih \
                           else ("W" if c.get("as_", 0) > c.get("hs", 0) else "L")

                pts = _si(p.get("PTS", 0)); fgm = _si(p.get("FGM", 0)); fga = _si(p.get("FGA", 0))
                fg3m= _si(p.get("FG3M", 0));fg3a= _si(p.get("FG3A", 0))
                ftm = _si(p.get("FTM", 0)); fta = _si(p.get("FTA", 0))
                oreb= _si(p.get("OREB", 0));dreb= _si(p.get("DREB", 0)); reb= _si(p.get("REB", 0))
                ast = _si(p.get("AST", 0)); stl = _si(p.get("STL", 0));  blk= _si(p.get("BLK", 0))
                tov = _si(p.get("TOV", 0)); pf  = _si(p.get("PF", 0));   pm = _si(p.get("PLUS_MINUS", 0))

                fgp = fgm / fga if fga > 0 else 0.0
                ftp = ftm / fta if fta > 0 else 0.0
                tsa = 2 * (fga + 0.44 * fta)
                ts  = pts / tsa if tsa > 0 else 0.0
                usg = (fga + 0.44 * fta + tov) / (mn / 5) if mn > 0 else 0.0
                fp  = pts + 1.25*reb + 1.5*ast + 2*stl + 2*blk - 0.5*tov
                b   = bio.get(pid, {})

                row = {
                    "PLAYER_ID": pid, "PLAYER_NAME": pname or b.get("PLAYER_NAME", ""),
                    "SEASON": "2025-26", "SEASON_TYPE": "Regular Season",
                    "PLAYER_POSITION":      b.get("PLAYER_POSITION", ""),
                    "PLAYER_POSITION_FULL": b.get("PLAYER_POSITION_FULL", ""),
                    "PLAYER_CURRENT_TEAM":  b.get("PLAYER_CURRENT_TEAM", ta),
                    "GAME_TEAM_ABBREVIATION": ta, "GAME_TEAM_NAME": b.get("GAME_TEAM_NAME", ""),
                    "GAME_ID": int(g), "GAME_DATE": date_str,
                    "MATCHUP": mu, "OPPONENT": opp, "IS_HOME": ih,
                    "WL": wl, "WL_WIN": 1 if wl == "W" else 0, "WL_LOSS": 1 if wl == "L" else 0,
                    "MIN": int(round(mn)), "MIN_NUM": round(mn, 1),
                    "FGM": fgm, "FGA": fga, "FG_PCT": round(fgp, 4),
                    "FG3M": fg3m, "FG3A": fg3a,
                    "FTM": ftm, "FTA": fta, "FT_PCT": round(ftp, 4),
                    "OREB": oreb, "DREB": dreb, "REB": reb, "AST": ast,
                    "STL": stl, "BLK": blk, "TOV": tov, "PF": pf,
                    "PTS": pts, "PLUS_MINUS": pm,
                    "TRUE_SHOOTING_PCT": round(ts, 4),
                    "USAGE_APPROX": round(usg, 2),
                    "FANTASY_PTS": round(fp, 2),
                    "SEASON_ID": 22025, "DNP": 0,
                }
                played_rows.append(row)
                results_map[_norm(pname)] = float(pts)

        except Exception as e:
            print(f"  ⚠ BoxScore parse game {g}: {e}")

    print(f"  Fetched {len(played_rows)} played rows, {len(players_in_box)} in box")

    if not played_rows:
        return _csv_fallback(date_str, played_rows, results_map, players_in_box)

    return played_rows, results_map, players_in_box


def _csv_fallback(date_str, played_rows, results_map, players_in_box):
    """Fallback: read today's scores from existing game log CSV."""
    if FILE_GL_2526.exists():
        try:
            df = pd.read_csv(FILE_GL_2526, parse_dates=["GAME_DATE"], low_memory=False)
            day = df[df["GAME_DATE"].dt.date == pd.Timestamp(date_str).date()]
            for _, r in day.iterrows():
                pnorm = _norm(str(r.get("PLAYER_NAME", "")))
                pts   = r.get("PTS")
                if pnorm and pts is not None:
                    try: results_map[pnorm] = float(pts)
                    except: pass
            print(f"  CSV fallback: {len(results_map)} player scores")
        except Exception as e:
            print(f"  ⚠ CSV fallback: {e}")
    return played_rows, results_map, players_in_box


# ── Step 2: Append game logs ──────────────────────────────────────────────────

def append_gamelogs(played_rows: list[dict], dnp_names: list[str], date_str: str) -> None:
    if not FILE_GL_2526.exists():
        print("  ⚠ gamelogs_2025_26.csv not found — cannot append")
        return
    try:
        df26 = pd.read_csv(FILE_GL_2526, parse_dates=["GAME_DATE"], low_memory=False)
        before = len(df26)
        if "DNP" not in df26.columns:
            df26["DNP"] = 0

        dnp_stubs = []
        for pname in dnp_names:
            stub = {c: np.nan for c in df26.columns}
            stub.update({"PLAYER_NAME": pname, "GAME_DATE": date_str,
                         "DNP": 1, "MIN_NUM": 0, "PTS": np.nan,
                         "SEASON": "2025-26", "SEASON_TYPE": "Regular Season", "SEASON_ID": 22025})
            dnp_stubs.append(stub)

        all_new = played_rows + dnp_stubs
        if not all_new:
            print("  No new rows to append"); return

        ndf = pd.DataFrame(all_new)
        ndf["GAME_DATE"] = pd.to_datetime(ndf["GAME_DATE"])
        if "DNP" not in ndf.columns: ndf["DNP"] = 0
        ndf = ndf.reindex(columns=df26.columns)

        updated = pd.concat([df26, ndf], ignore_index=True)
        updated["GAME_DATE"] = pd.to_datetime(updated["GAME_DATE"])
        updated = updated.sort_values(["PLAYER_NAME", "GAME_DATE"]).reset_index(drop=True)
        updated = updated.drop_duplicates(subset=["PLAYER_NAME", "GAME_DATE"], keep="last")

        # Recompute rolling for affected players
        new_players = {r.get("PLAYER_NAME", "") for r in played_rows if r.get("PLAYER_NAME")}
        dnp_players = set(filter(None, dnp_names))
        all_players = new_players | dnp_players
        if all_players:
            updated = recompute_rolling(updated, all_players)

        updated.to_csv(FILE_GL_2526, index=False)
        print(f"  ✓ Game log: {before:,} → {len(updated):,} rows "
              f"(+{len(played_rows)} played, +{len(dnp_stubs)} DNP) "
              f"| Rolling recomputed: {len(all_players)} players")
    except Exception as e:
        print(f"  ⚠ append_gamelogs: {e}")


# ── Step 3: Grade plays ───────────────────────────────────────────────────────

def grade_plays(
    plays: list[dict],
    results_map: dict[str, float],
    players_in_box: set[str],
    date_str: str,
    played_rows: list[dict] | None = None,
) -> tuple[list[dict], int, int, int]:
    box_stats: dict[str, dict] = {}
    if played_rows:
        for row in played_rows:
            pname = str(row.get("PLAYER_NAME", ""))
            if pname:
                box_stats[_norm(pname)] = row

    for p in plays:
        if p.get("date") != date_str:
            continue
        if p.get("result") in ("WIN", "LOSS", "DNP", "PUSH"):
            continue

        pname = p.get("player", "")
        line  = float(p.get("line", 0))
        dir_  = str(p.get("dir", p.get("direction", ""))).upper().replace("LEAN ", "")

        pnorm      = _norm(pname)
        actual_pts = results_map.get(pnorm)
        if actual_pts is None:
            actual_pts = resolve_grade_name(pname, results_map)

        if actual_pts is None:
            p["result"]           = "DNP"
            p["actualPts"]        = None
            p["lossType"]         = None
            p["postMatchReason"]  = "🚫 No score found — DNP or name mismatch."
            continue

        if abs(actual_pts - line) < 0.05:
            result = "PUSH"
        elif "OVER" in dir_:
            result = "WIN" if actual_pts > line else "LOSS"
        elif "UNDER" in dir_:
            result = "WIN" if actual_pts <= line else "LOSS"
        else:
            result = "WIN" if actual_pts > line else "LOSS"

        p["result"]    = result
        p["actualPts"] = actual_pts
        p["delta"]     = round(actual_pts - line, 1)

        brow = box_stats.get(pnorm, {})
        box_data = {
            "actual_pts": actual_pts,
            "actual_min": float(brow.get("MIN_NUM", 0) or 0),
            "actual_fga": float(brow.get("FGA", 0) or 0),
            "actual_fgm": float(brow.get("FGM", 0) or 0),
        }
        post, loss_type = generate_post_match_reason(p, box_data)
        p["lossType"]        = loss_type if result == "LOSS" else None
        p["postMatchReason"] = post
        p["actual_min"]      = box_data["actual_min"]
        p["actual_fga"]      = box_data["actual_fga"]
        p["actual_fgm"]      = box_data["actual_fgm"]

    wins = losses = dnps = 0
    for p in plays:
        if p.get("date") != date_str: continue
        r = p.get("result")
        if r == "WIN":    wins   += 1
        elif r == "LOSS": losses += 1
        elif r == "DNP":  dnps   += 1

    return plays, wins, losses, dnps


# ── Step 4: Season JSON update ────────────────────────────────────────────────

def update_season_json(graded: list[dict], date_str: str) -> None:
    existing: list[dict] = []
    if FILE_SEASON_2526.exists():
        try:
            with open(FILE_SEASON_2526) as f:
                existing = json.load(f)
        except Exception:
            pass

    by_key: dict[tuple, dict] = {
        (p.get("player"), p.get("date")): p
        for p in existing if p.get("result") in ("WIN", "LOSS", "DNP", "PUSH")
    }
    for p in graded:
        if p.get("date") == date_str:
            by_key[(p.get("player"), p.get("date"))] = p

    other = [p for p in existing if p.get("date") != date_str]
    merged = other + list(by_key.values())
    merged.sort(key=lambda p: (p.get("date", ""), p.get("player", "")))

    FILE_SEASON_2526.parent.mkdir(parents=True, exist_ok=True)
    with open(FILE_SEASON_2526, "w") as f:
        json.dump(clean_json(merged), f, indent=2)
    print(f"  ✓ season_2025_26.json → {len(merged):,} plays")


# ── DVP rebuild ───────────────────────────────────────────────────────────────

def rebuild_dvp() -> None:
    try:
        if not FILE_GL_2526.exists():
            return
        from rolling_engine import build_dynamic_dvp, filter_played
        df = pd.read_csv(FILE_GL_2526, parse_dates=["GAME_DATE"], low_memory=False)
        played = filter_played(df)
        dvp = build_dynamic_dvp(played)
        FILE_DVP.parent.mkdir(parents=True, exist_ok=True)
        with open(FILE_DVP, "w") as f:
            json.dump(dvp, f, indent=2)
        print(f"  ✓ DVP rebuilt ({len(dvp)} team-pos pairs)")
    except Exception as e:
        print(f"  ⚠ DVP rebuild: {e}")


# ── Main ──────────────────────────────────────────────────────────────────────

def run_grade(date_str: str | None = None, push: bool = True) -> None:
    from config import DATA_DIR
    if date_str is None:
        date_str = (now_uk() - timedelta(days=1)).strftime("%Y-%m-%d")

    lock = DATA_DIR / ".b0.lock"
    if lock.exists():
        age = __import__("time").time() - lock.stat().st_mtime
        if age < 1800:
            print(f"  ⚠ B0 lock found ({age:.0f}s old) — skipping. Delete {lock} if stale.")
            return
    lock.write_text(str(__import__("time").time()))

    try:
        _run_grade_locked(date_str, push)
    finally:
        lock.unlink(missing_ok=True)


def _run_grade_locked(date_str: str, push: bool) -> None:
    print(f"\n  {VERSION_TAG} — B0 Grade  [{date_str}]")

    # [1] Box scores
    print("[1/5] Fetching box scores...")
    played_rows, results_map, players_in_box = fetch_box_scores(date_str)
    if not results_map:
        print(f"  No scores for {date_str}. Skipping.")
        return

    # [2] Load plays
    print("[2/5] Loading plays...")
    plays: list[dict] = []
    if FILE_TODAY.exists():
        try:
            with open(FILE_TODAY) as f: plays = json.load(f)
        except Exception: pass

    # Also pull ungraded plays from season JSON
    if FILE_SEASON_2526.exists():
        try:
            with open(FILE_SEASON_2526) as f: season_plays = json.load(f)
            existing_keys = {(p.get("player"), p.get("date"), str(p.get("line"))) for p in plays}
            for p in season_plays:
                if p.get("date") == date_str:
                    k = (p.get("player"), p.get("date"), str(p.get("line")))
                    if k not in existing_keys:
                        plays.append(p)
        except Exception: pass

    to_grade = [p for p in plays if p.get("date") == date_str
                and p.get("result") not in ("WIN", "LOSS", "DNP", "PUSH")]
    print(f"  Plays to grade: {len(to_grade)} | Total loaded: {len(plays)}")

    wins = losses = dnps = 0
    graded_today: list[dict] = []

    # [3] Grade
    if to_grade:
        print("[3/5] Grading...")
        plays, wins, losses, dnps = grade_plays(plays, results_map, players_in_box, date_str, played_rows)
        total = wins + losses + dnps
        hr = f"{wins/total*100:.1f}%" if total > 0 else "—"
        print(f"  Graded: {total} | W:{wins} L:{losses} DNP:{dnps} | HR:{hr}")

        DATA_DIR.mkdir(parents=True, exist_ok=True)
        with open(FILE_TODAY, "w") as f:
            json.dump(clean_json(plays), f, indent=2)

        graded_today = [p for p in plays if p.get("date") == date_str
                        and p.get("result") in ("WIN", "LOSS", "DNP", "PUSH")]
        update_season_json(graded_today, date_str)

        # Monthly split update
        try:
            month_str = date_str[:7]
            update_month(graded_today, "2025_26", month_str)
            print(f"  ✓ Monthly file updated: {month_str}")
        except Exception as e:
            print(f"  ⚠ Monthly update: {e}")
    else:
        print("  Nothing to grade — game log will still be updated.")

    # [4] Append game logs
    print("[4/5] Appending game logs...")
    dnp_names = [p.get("player", "") for p in graded_today if p.get("result") == "DNP"]
    if played_rows:
        append_gamelogs(played_rows, dnp_names, date_str)
    else:
        print("  No played rows — game log unchanged.")

    # [5] Rebuild caches
    print("[5/5] Rebuilding DVP...")
    rebuild_dvp()

    if push:
        from git_push import push as git_push
        from monthly_split import get_push_paths
        month_str = date_str[:7]
        extra = get_push_paths("2025_26", only_current_month=True)
        git_push(f"V2.0 grade {date_str} — {wins}W/{losses}L", grade=True)

    if to_grade:
        print(f"\n  B0 complete. {date_str}  {wins}W / {losses}L / {dnps} DNP")
    else:
        print(f"\n  B0 complete. {date_str}  (game log updated, no plays graded)")


if __name__ == "__main__":
    override = sys.argv[1] if len(sys.argv) > 1 else None
    run_grade(override)
