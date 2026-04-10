"""
PropEdge V2.0 — generate_season_json.py
Full rebuild of both season JSONs from real prop lines + game logs.
Grades 2024-25 retroactively using the game log CSVs.

RUN: python3 run.py generate
"""
from __future__ import annotations

import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
ROOT = Path(__file__).parent.resolve()
sys.path.insert(0, str(ROOT))

from config import (
    VERSION_TAG, FILE_GL_2425, FILE_GL_2526, FILE_H2H,
    FILE_PROPS_2425, FILE_PROPS_2526,
    FILE_SEASON_2425, FILE_SEASON_2526, FILE_DVP,
    MIN_PRIOR_GAMES, assign_tier, clean_json, TIER_STAKES,
    now_uk,
)
from player_name_aliases import _norm, resolve_name
from rolling_engine import (
    filter_played, build_player_index, get_prior_games,
    build_dynamic_dvp, build_pace_rank, build_opp_def_caches,
    build_rest_days_map, extract_features,
)
from reasoning_engine import generate_pre_match_reason, generate_post_match_reason
from monthly_split import write_monthly_split, verify_monthly_integrity
from model_engine import predict as model_predict


# ── Load prop lines from Excel ────────────────────────────────────────────────

def load_props(source_file: Path, season_start: str, season_end: str, season_label: str) -> list[dict]:
    if not source_file.exists():
        print(f"  ✗ Missing: {source_file.name}")
        return []
    s = pd.Timestamp(season_start); e = pd.Timestamp(season_end)
    try:
        # Try Player_Points_Props sheet first, fall back to first sheet
        try:
            xl = pd.read_excel(source_file, sheet_name="Player_Points_Props")
        except Exception:
            xl = pd.read_excel(source_file)

        xl["Date"] = pd.to_datetime(xl["Date"], errors="coerce")
        xl = xl[(xl["Date"] >= s) & (xl["Date"] <= e)].dropna(subset=["Line"])

        if xl.empty:
            print(f"  ✗ No rows in range {season_start}→{season_end} in {source_file.name}")
            return []

        props = []
        for _, r in xl.iterrows():
            try:
                home = str(r.get("Home") or r.get("home") or "").strip()
                away = str(r.get("Away") or r.get("away") or "").strip()
                game = str(r.get("Game") or r.get("matchup") or f"{away} @ {home}").strip()
                props.append({
                    "player":       str(r.get("Player") or r.get("player_name", "")).strip(),
                    "date":         str(r["Date"].date()),
                    "line":         float(r["Line"]),
                    "over_odds":    float(r.get("Over Odds") or r.get("over_odds") or -110),
                    "under_odds":   float(r.get("Under Odds") or r.get("under_odds") or -110),
                    "books":        int(r.get("Books") or r.get("books") or 0),
                    "min_line":     float(r["Min Line"]) if pd.notna(r.get("Min Line")) else None,
                    "max_line":     float(r["Max Line"]) if pd.notna(r.get("Max Line")) else None,
                    "game":         game,
                    "home":         home,
                    "away":         away,
                    "game_time_et": str(r.get("Game_Time_ET") or r.get("game_time") or "").strip(),
                })
            except Exception:
                continue
        print(f"  Props ({season_label}): {len(props):,} rows from {source_file.name}")
        return props
    except Exception as e:
        print(f"  ✗ Excel read error ({source_file.name}): {e}")
        return []


# ── Score and grade one play ──────────────────────────────────────────────────

def score_and_grade(
    props: list[dict],
    pidx: dict,
    played: pd.DataFrame,
    combined_df: pd.DataFrame,
    h2h_lkp: dict,
    dvp: dict, pace: dict, otr: dict, ovr: dict, rmap: dict,
    season: str,
    prev_directions: dict | None = None,
) -> list[dict]:

    from model_engine import load_model, load_thresholds
    from batch_predict import _count_flags, _flag_details, _american_to_prob

    model, cal, feat_cols = load_model()
    thresholds = load_thresholds()
    nmap = {_norm(k): k for k in pidx}

    scored: list[dict] = []
    skipped = 0
    total   = len(props)

    for pi, prop in enumerate(props):
        if pi % 2000 == 0 and pi > 0:
            print(f"    {pi:,}/{total:,} | scored={len(scored)} skipped={skipped}")

        line       = float(prop["line"])
        player_raw = prop["player"]
        date_str   = prop["date"]
        home_team  = str(prop.get("home", "")).upper()
        away_team  = str(prop.get("away", "")).upper()
        game       = str(prop.get("game", ""))
        gtime      = str(prop.get("game_time_et", ""))

        player = resolve_name(player_raw, nmap)
        if player is None:
            skipped += 1; continue

        prior = get_prior_games(pidx, player, date_str)
        if len(prior) < MIN_PRIOR_GAMES:
            skipped += 1; continue

        ptm = str(prior["GAME_TEAM_ABBREVIATION"].iloc[-1]).upper() \
              if "GAME_TEAM_ABBREVIATION" in prior.columns else ""
        is_home = (ptm == home_team) if ptm and home_team else None
        opp     = home_team if ptm == away_team else away_team if ptm == home_team else ""
        pos_raw = str(prior["PLAYER_POSITION"].iloc[-1]) if "PLAYER_POSITION" in prior.columns else "G"
        rd      = rmap.get((player, date_str), 2)
        h2h_row = h2h_lkp.get((_norm(player), opp.upper()))

        f = extract_features(
            prior=prior, line=line, opponent=opp, rest_days=rd,
            pos_raw=pos_raw, game_date=pd.Timestamp(date_str),
            min_line=prop.get("min_line"), max_line=prop.get("max_line"),
            dyn_dvp=dvp, pace_rank=pace, opp_trend=otr, opp_var=ovr,
            is_home=is_home, h2h_row=h2h_row,
        )
        if f is None:
            skipped += 1; continue

        # Market features
        f["books_log"]         = float(np.log1p(prop.get("books", 1)))
        f["implied_over_prob"] = _american_to_prob(prop.get("over_odds", -110))
        f["implied_under_prob"]= _american_to_prob(prop.get("under_odds", -110))

        # V2.0 model inference
        from config import NEVER_USE_AS_FEATURES
        safe = [c for c in feat_cols if c not in NEVER_USE_AS_FEATURES]
        row  = {c: f.get(c, 0.0) or 0.0 for c in safe}
        X    = pd.DataFrame([row])[safe].fillna(0).values
        raw_p = float(model.predict_proba(X)[0, 1])
        p_over = float(np.clip(cal.predict([raw_p])[0], 0.01, 0.99))
        p_under = 1.0 - p_over

        # Direction — locked for already-graded plays
        _locked = (prev_directions or {}).get((player, date_str))
        if _locked:
            dl = _locked
        else:
            dl = "OVER" if p_over >= 0.5 else "UNDER"

        dir_conf = p_over if dl == "OVER" else p_under
        tier     = assign_tier(dir_conf, thresholds)

        pred_pts = round(p_over * f.get("L10", line) + p_under * f.get("L30", line), 1)

        # Grade if game log has result
        result    = ""; actual_pts = None; delta = None
        actual_row = played[
            (played["PLAYER_NAME"] == player) &
            (played["GAME_DATE"] == pd.Timestamp(date_str))
        ]
        if not actual_row.empty:
            actual_pts = float(actual_row["PTS"].iloc[0])
            if abs(actual_pts - line) < 0.05:
                result = "PUSH"
            elif dl == "OVER":
                result = "WIN" if actual_pts > line else "LOSS"
            else:
                result = "WIN" if actual_pts <= line else "LOSS"
            delta = round(actual_pts - line, 1)
        elif pd.Timestamp(date_str) < pd.Timestamp("today"):
            date_games = combined_df[combined_df["GAME_DATE"] == pd.Timestamp(date_str)]
            if not date_games.empty:
                result = "DNP"

        # Recent 20
        pts_vals   = prior["PTS"].values[-20:]
        dates_vals = prior["GAME_DATE"].values[-20:]
        home_vals  = prior["IS_HOME"].values[-20:] if "IS_HOME" in prior.columns else [True] * len(pts_vals)
        hr10 = float((pts_vals > line).mean()) if len(pts_vals) > 0 else 0.5

        play = {
            "player":       player,     "date":        date_str,
            "match":        game or f"{away_team} @ {home_team}",
            "fullMatch":    game or f"{away_team} @ {home_team}",
            "game":         game,       "home":        home_team,
            "away":         away_team,  "opponent":    opp,
            "position":     pos_raw,    "isHome":      is_home,
            "gameTime":     gtime,      "game_time":   gtime,
            "team":         ptm,        "ptm":         ptm,
            "season":       season,
            "line":         line,
            "overOdds":     prop.get("over_odds", -110),
            "underOdds":    prop.get("under_odds", -110),
            "books":        prop.get("books", 0),
            "min_line":     prop.get("min_line"),
            "max_line":     prop.get("max_line"),
            "lineHistory":  [],
            # Model outputs
            "p_over":           round(p_over, 4),   "p_under":       round(p_under, 4),
            "dir":              dl,                  "direction":     dl,
            "conf":             round(dir_conf, 4),  "confidence_pct":round(dir_conf*100,1),
            "tier":             tier,                "elite_tier":    tier,
            "elite_prob":       round(dir_conf, 4),
            "elite_stake":      TIER_STAKES.get(tier, 0.0),
            "tierLabel":        tier,                "units":         TIER_STAKES.get(tier,0.0),
            "predPts":          pred_pts,            "predGap":       round(pred_pts-line,2),
            "predicted_pts":    pred_pts,
            "calProb":          round(p_over, 4),    "v12_clf_prob":  round(p_over, 4),
            "flags":            _count_flags(f, dl), "flagsStr":      f"{_count_flags(f,dl)}/10",
            "flagDetails":      _flag_details(f, dl),
            "all_clf_agree":    bool(f.get("all_windows_over") or f.get("all_windows_under")),
            "reg_consensus":    bool(f.get("all_windows_over") or f.get("all_windows_under")),
            "v12_extreme":      bool(dir_conf >= 0.95),
            "trust_mean":       0.70,
            "q25_v12":          round(line - f.get("std10", 5), 1),
            "q75_v12":          round(line + f.get("std10", 5), 1),
            "q_confidence":     round(dir_conf, 3),
            "real_gap_v92":     round(f.get("L30", 0) - line, 2),
            "real_gap_v12":     round(pred_pts - line, 2),
            # Rolling
            "l3":   round(f.get("L3", 0), 1),  "l5":  round(f.get("L5", 0), 1),
            "l10":  round(f.get("L10",0), 1),  "l20": round(f.get("L20",0), 1),
            "l30":  round(f.get("L30",0), 1),  "std10": round(f.get("std10",0),1),
            "hr10": round(hr10, 3),             "hr30": round(hr10, 3),
            "volume": round(f.get("volume",0),1), "trend": round(f.get("trend",0),1),
            "momentum": round(f.get("momentum",0),1),
            # Minutes
            "min_l10": round(f.get("min_l10",28),1), "minL10": round(f.get("min_l10",28),1),
            "min_l30": round(f.get("min_l30",28),1),
            "usage_l10": round(f.get("usage_l10",0.18),3),
            "fta_l10": round(f.get("fta_l10",0),1), "fga_l10": round(f.get("fga_l10",0),1),
            "pts_per_min": round(f.get("pts_per_min",0.5),3),
            "home_away_split": round(f.get("home_away_split",0),1),
            "homeAvgPts": round(f.get("home_l10",f.get("L10",0)),1),
            "awayAvgPts": round(f.get("away_l10",f.get("L10",0)),1),
            "level_ewm": round(f.get("level_ewm",f.get("L10",0)),1),
            "line_vs_l30": round(f.get("line_vs_l30",0),2),
            # Context
            "is_b2b": bool(f.get("is_b2b",0)), "rest_days": int(rd),
            "defP": int(f.get("defP_dynamic",15)), "defP_dynamic": int(f.get("defP_dynamic",15)),
            "pace_rank": int(f.get("pace_rank",15)), "pace": int(f.get("pace_rank",15)),
            "seasonProgress": round(f.get("season_progress",0.5),3),
            "meanReversionRisk": round(f.get("mean_reversion_risk",0),2),
            "extreme_hot": bool(f.get("extreme_hot",False)),
            "extreme_cold": bool(f.get("extreme_cold",False)),
            "is_long_rest": bool(f.get("is_long_rest",False)),
            "earlySeasonW": round(f.get("early_season_weight",1),3),
            # H2H
            "h2hG": int(f.get("h2h_games",0)), "h2h_games": int(f.get("h2h_games",0)),
            "h2h":  round(f.get("h2h_avg"),1) if f.get("h2h_avg") is not None else None,
            "h2h_avg": round(f.get("h2h_avg"),1) if f.get("h2h_avg") is not None else None,
            "h2hTsDev": round(f.get("h2h_ts_dev",0),2),
            "h2hFgaDev": round(f.get("h2h_fga_dev",0),2),
            "h2hConfidence": round(f.get("h2h_conf",0),3),
            # Recent
            "recent20":      [float(v) for v in pts_vals],
            "recent20dates": [str(pd.Timestamp(d).date()) for d in dates_vals],
            "recent20homes": [bool(v) for v in home_vals],
            # Grade
            "result":          result,      "actualPts":       actual_pts,
            "delta":           delta,       "lossType":        None,
            "preMatchReason":  "",          "postMatchReason": None,
            "source":          "excel",
        }

        play["preMatchReason"] = generate_pre_match_reason(play)

        if result in ("WIN", "LOSS") and actual_pts is not None:
            r0 = actual_row.iloc[0] if not actual_row.empty else pd.Series()
            box_data = {
                "actual_pts": actual_pts,
                "actual_min": float(r0.get("MIN_NUM", 0) or 0),
                "actual_fga": float(r0.get("FGA", 0) or 0),
                "actual_fgm": float(r0.get("FGM", 0) or 0),
            }
            post, loss_type = generate_post_match_reason(play, box_data)
            play["postMatchReason"] = post
            play["lossType"]        = loss_type if result == "LOSS" else None
            play["actual_min"]      = box_data["actual_min"]
            play["actual_fga"]      = box_data["actual_fga"]
            play["actual_fgm"]      = box_data["actual_fgm"]
        elif result == "PUSH":
            play["postMatchReason"] = f"PUSH — scored exactly {actual_pts:.0f}."

        scored.append(play)

    print(f"  Scored: {len(scored):,} | Skipped: {skipped:,}")
    return scored


def _print_stats(label: str, plays: list[dict]) -> None:
    total  = len(plays)
    graded = [p for p in plays if p.get("result") in ("WIN", "LOSS")]
    wins   = [p for p in graded if p.get("result") == "WIN"]
    hr = f"{len(wins)/len(graded)*100:.1f}%" if graded else "—"
    grade_pct = len(graded)/total*100 if total else 0
    flag = " ⚠ Low graded rate" if grade_pct < 40 else ""
    print(f"  ✓ {label}: {total:,} plays | {len(graded):,} graded (HR:{hr}){flag}")

    # Tier breakdown
    for tier in ["APEX", "ULTRA", "ELITE", "STRONG", "PLAY"]:
        tp = [p for p in graded if p.get("tier") == tier or p.get("elite_tier") == tier]
        if tp:
            tw = sum(1 for p in tp if p.get("result") == "WIN")
            print(f"    {tier:<8} {len(tp):5,} plays  {tw/len(tp)*100:.1f}%")


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    print(f"\n  {VERSION_TAG} — generate_season_json.py")

    lock = FILE_SEASON_2526.parent / ".generate.lock"
    if lock.exists():
        age = time.time() - lock.stat().st_mtime
        if age < 3600:
            print(f"  ✗ Already running (lock {age:.0f}s old). Delete {lock} if stale.")
            return
    lock.write_text(str(time.time()))
    try:
        _run_generate()
    finally:
        lock.unlink(missing_ok=True)


def _run_generate() -> None:
    import shutil

    # Backup existing JSONs
    for fp in (FILE_SEASON_2425, FILE_SEASON_2526):
        if fp.exists():
            shutil.copy2(fp, fp.with_suffix(".bak"))
            print(f"  ✓ Backed up {fp.name}")

    # [1] Game logs
    print("\n[1/5] Loading game logs...")
    dfs = []
    for fp in (FILE_GL_2425, FILE_GL_2526):
        if fp.exists():
            try: dfs.append(pd.read_csv(fp, parse_dates=["GAME_DATE"], low_memory=False))
            except Exception as e: print(f"  ⚠ {fp.name}: {e}")
        else:
            print(f"  ⚠ Missing: {fp.name}")
    if not dfs:
        print("  ✗ No game logs. Place CSVs in source-files/ and retry."); return

    combined = pd.concat(dfs, ignore_index=True)
    played   = filter_played(combined)
    pidx     = build_player_index(played)
    print(f"  {len(played):,} played rows | {len(pidx):,} players")

    # [2] Caches
    print("\n[2/5] Building caches...")
    dvp      = build_dynamic_dvp(played)
    pace     = build_pace_rank(played)
    otr, ovr = build_opp_def_caches(played)
    rmap     = build_rest_days_map(played)

    # Save DVP
    FILE_DVP.parent.mkdir(parents=True, exist_ok=True)
    with open(FILE_DVP, "w") as f: json.dump(dvp, f, indent=2)
    print(f"  ✓ DVP saved ({len(dvp)} pairs)")

    # [3] H2H
    print("\n[3/5] Loading H2H...")
    h2h_lkp: dict = {}
    if FILE_H2H.exists():
        try:
            dfh = pd.read_csv(FILE_H2H, low_memory=False)
            h2h_lkp = {
                (_norm(str(r.get("PLAYER_NAME",""))), str(r.get("OPPONENT","")).strip().upper()): r.to_dict()
                for _, r in dfh.iterrows()
            }
            print(f"  H2H pairs: {len(h2h_lkp):,}")
        except Exception as e: print(f"  ⚠ H2H: {e}")
    else:
        print("  ⚠ h2h_database.csv not found")

    kw = dict(pidx=pidx, played=played, combined_df=combined,
              h2h_lkp=h2h_lkp, dvp=dvp, pace=pace, otr=otr, ovr=ovr, rmap=rmap)

    # [4] 2024-25
    print("\n[4/5] 2024-25 season...")
    props_2425 = load_props(FILE_PROPS_2425, "2024-10-01", "2025-09-30", "2024-25")
    plays_2425: list[dict] = []
    if props_2425:
        # Lock directions from existing JSON
        prev_2425: dict[tuple, str] = {}
        if FILE_SEASON_2425.exists():
            try:
                prev = json.loads(FILE_SEASON_2425.read_text())
                prev_2425 = {(p.get("player",""),p.get("date","")): p.get("dir","OVER")
                             for p in prev if p.get("result") in ("WIN","LOSS","DNP","PUSH")}
                print(f"  Locked {len(prev_2425):,} directions from existing 2024-25 JSON")
            except Exception: pass

        plays_2425 = score_and_grade(props_2425, season="2024-25", prev_directions=prev_2425, **kw)
        plays_2425.sort(key=lambda p: (p.get("date",""), p.get("player","")))
        FILE_SEASON_2425.parent.mkdir(parents=True, exist_ok=True)
        with open(FILE_SEASON_2425, "w") as f:
            json.dump(clean_json(plays_2425), f, indent=2)
        _print_stats("2024-25", plays_2425)

        counts = write_monthly_split(plays_2425, "2024_25")
        ok, msg = verify_monthly_integrity("2024_25", plays_2425)
        print(f"  {'✓' if ok else '⚠'} Monthly 2024-25: {len(counts)} files | {msg}")
    else:
        print("  ✗ No 2024-25 props — ensure Player_lines_2024-25.xlsx in source-files/")

    # [5] 2025-26
    print("\n[5/5] 2025-26 season...")
    props_2526 = load_props(FILE_PROPS_2526, "2025-10-01", "2026-09-30", "2025-26")
    plays_2526: list[dict] = []
    if props_2526:
        prev_2526: dict[tuple, str] = {}
        if FILE_SEASON_2526.exists():
            try:
                prev = json.loads(FILE_SEASON_2526.read_text())
                prev_2526 = {(p.get("player",""),p.get("date","")): p.get("dir","OVER")
                             for p in prev if p.get("result") in ("WIN","LOSS","DNP","PUSH")}
                print(f"  Locked {len(prev_2526):,} directions from existing 2025-26 JSON")
            except Exception: pass

        plays_2526 = score_and_grade(props_2526, season="2025-26", prev_directions=prev_2526, **kw)
        plays_2526.sort(key=lambda p: (p.get("date",""), p.get("player","")))
        FILE_SEASON_2526.parent.mkdir(parents=True, exist_ok=True)
        with open(FILE_SEASON_2526, "w") as f:
            json.dump(clean_json(plays_2526), f, indent=2)
        _print_stats("2025-26", plays_2526)

        counts = write_monthly_split(plays_2526, "2025_26")
        ok, msg = verify_monthly_integrity("2025_26", plays_2526)
        print(f"  {'✓' if ok else '⚠'} Monthly 2025-26: {len(counts)} files | {msg}")
    else:
        print("  ✗ No 2025-26 props — ensure Player_lines_2025-26.xlsx in source-files/")

    # Push
    try:
        from git_push import push
        from monthly_split import get_push_paths
        all_paths = (get_push_paths("2024_25", only_current_month=False) +
                     get_push_paths("2025_26", only_current_month=False))
        push(f"V2.0 generate {now_uk().strftime('%Y-%m-%d %H:%M UTC')}",
             files=all_paths + ["data/dvp_rankings.json"])
        print("  ✓ Pushed to GitHub Pages")
    except Exception as e:
        print(f"  ⚠ Push: {e}")

    print(f"\n  ✓ Generate complete.")
    all_plays = plays_2425 + plays_2526
    if all_plays:
        graded = [p for p in all_plays if p.get("result") in ("WIN","LOSS")]
        wins   = sum(1 for p in graded if p.get("result")=="WIN")
        print(f"  Combined: {len(all_plays):,} plays | {len(graded):,} graded | HR:{wins/len(graded)*100:.1f}%" if graded else "")


if __name__ == "__main__":
    main()
