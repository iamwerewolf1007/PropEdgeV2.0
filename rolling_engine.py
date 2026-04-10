"""
PropEdge V2.0 — rolling_engine.py
Reads pre-computed rolling columns from game log CSVs (same pattern as V16).
Used by batch_predict.py and generate_season_json.py for feature extraction.

The game log CSVs contain pre-computed rolling stats (L3_PTS, L5_PTS, etc.)
built with shift(1) — "form entering each game". For live predictions we
compute from raw pts_arr to avoid the rolling-staleness bug.
"""
from __future__ import annotations

import math
import numpy as np
import pandas as pd

from config import get_pos_group, MIN_PRIOR_GAMES, H2H_MIN_GAMES


# ── Safe helpers ──────────────────────────────────────────────────────────────

def _f(row, col, default=0.0):
    v = row.get(col) if isinstance(row, dict) else (row[col] if col in row.index else default)
    try:
        f = float(v)
        return default if (math.isnan(f) or math.isinf(f)) else f
    except (TypeError, ValueError):
        return default

def _sm(arr, n):
    sl = arr[-n:] if len(arr) >= n else arr
    return float(np.mean(sl)) if len(sl) > 0 else 0.0

def _hr(arr, line, n):
    sl = arr[-n:] if len(arr) >= n else arr
    return float((np.array(sl) > line).mean()) if len(sl) > 0 else 0.5


# ── Filter + index ────────────────────────────────────────────────────────────

def filter_played(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["GAME_DATE"] = pd.to_datetime(df["GAME_DATE"], errors="coerce")
    mask = (df["MIN_NUM"].fillna(0) > 0) & df["PTS"].notna()
    return df[mask].copy()


def build_player_index(played: pd.DataFrame) -> dict:
    idx = {}
    for name, grp in played.groupby("PLAYER_NAME"):
        idx[name] = grp.sort_values("GAME_DATE").reset_index(drop=True)
    return idx


def get_prior_games(player_idx: dict, player: str, game_date: str) -> pd.DataFrame:
    if player not in player_idx:
        return pd.DataFrame()
    grp = player_idx[player]
    gd  = pd.Timestamp(game_date)
    return grp[grp["GAME_DATE"] < gd].copy()


# ── DVP / pace / rest caches ──────────────────────────────────────────────────

def build_dynamic_dvp(played: pd.DataFrame) -> dict:
    if "OPPONENT" not in played.columns or "PTS" not in played.columns:
        return {}
    pos_col = "PLAYER_POSITION" if "PLAYER_POSITION" in played.columns else None
    recent  = played[played["GAME_DATE"] >= played["GAME_DATE"].max() - pd.Timedelta(days=30)]
    if len(recent) < 100:
        recent = played
    recent = recent.copy()
    recent["_pg"] = recent[pos_col].fillna("G").apply(get_pos_group) if pos_col else "Guard"
    avg_pts = recent.groupby(["OPPONENT", "_pg"])["PTS"].mean()
    result  = {}
    for pos in ["Guard", "Forward", "Center"]:
        try:
            sub  = avg_pts.xs(pos, level="_pg")
            rank = sub.rank(ascending=True, method="min").astype(int)
            for team, r in rank.items():
                result[f"{team}|{pos}"] = int(r)
        except KeyError:
            pass
    return result


def build_pace_rank(played: pd.DataFrame) -> dict:
    if "GAME_TEAM_ABBREVIATION" not in played.columns:
        return {}
    recent = played[played["GAME_DATE"] >= played["GAME_DATE"].max() - pd.Timedelta(days=30)]
    if recent.empty:
        recent = played
    totals = recent.groupby("GAME_TEAM_ABBREVIATION")["MIN_NUM"].count()
    return totals.rank(ascending=False, method="min").astype(int).to_dict()


def build_opp_def_caches(played: pd.DataFrame) -> tuple[dict, dict]:
    if "OPPONENT" not in played.columns:
        return {}, {}
    pos_col = "PLAYER_POSITION" if "PLAYER_POSITION" in played.columns else None
    played  = played.copy()
    played["_pg"] = played[pos_col].fillna("G").apply(get_pos_group) if pos_col else "Guard"
    trend_d: dict = {}; var_d: dict = {}
    for (opp, pos), grp in played.groupby(["OPPONENT", "_pg"]):
        pts_arr = grp.sort_values("GAME_DATE")["PTS"].values[-20:].astype(float)
        if len(pts_arr) >= 5:
            trend_d[f"{opp}|{pos}"] = float(pts_arr[-5:].mean() - pts_arr.mean())
            var_d[f"{opp}|{pos}"]   = float(np.std(pts_arr))
    return trend_d, var_d


def build_rest_days_map(played: pd.DataFrame) -> dict:
    result = {}
    for name, grp in played.groupby("PLAYER_NAME"):
        grp   = grp.sort_values("GAME_DATE").drop_duplicates("GAME_DATE")
        dates = grp["GAME_DATE"].values
        diffs = pd.Series(dates).diff().dt.days.fillna(7).clip(upper=14).astype(int)
        for gd, rd in zip(dates, diffs):
            result[(name, str(pd.Timestamp(gd).date()))] = int(rd)
    return result


# ── Feature extraction ────────────────────────────────────────────────────────

def extract_features(
    prior:     pd.DataFrame,
    line:      float,
    opponent:  str,
    rest_days: int,
    pos_raw:   str,
    game_date: pd.Timestamp,
    min_line:  float | None,
    max_line:  float | None,
    dyn_dvp:   dict,
    pace_rank: dict,
    opp_trend: dict,
    opp_var:   dict,
    is_home:   bool | None = None,
    h2h_row:   dict | None = None,
) -> dict | None:
    if len(prior) < MIN_PRIOR_GAMES:
        return None

    last = prior.iloc[-1]

    def col(c, d=0.0):
        return _f(last, c, d)

    # ── Rolling PTS — computed from raw pts array (avoids staleness bug) ────
    pts_arr = prior["PTS"].values.astype(float)
    n       = len(pts_arr)
    std10   = max(float(np.std(pts_arr[-10:])) if n >= 2 else 1.0, 0.5)
    hr10    = _hr(pts_arr, line, 10)
    hr30    = _hr(pts_arr, line, 30)
    L3      = _sm(pts_arr, 3)
    L5      = _sm(pts_arr, 5)
    L10     = _sm(pts_arr, 10)
    L20     = _sm(pts_arr, 20)
    L30     = _sm(pts_arr, 30)

    # ── Minutes ──────────────────────────────────────────────────────────────
    min_l10  = col("L10_MIN_NUM", 28.0)
    min_l30  = col("L30_MIN_NUM", 28.0)
    min_l3   = col("L3_MIN_NUM",  28.0)
    mins_arr = prior["MIN_NUM"].fillna(0).values.astype(float)
    min_cv   = float(np.std(mins_arr[-10:]) / max(min_l10, 1.0)) if len(mins_arr) >= 3 else 0.2
    recent_min_trend = float(_sm(mins_arr, 5) - min_l10)

    # ── Shooting ─────────────────────────────────────────────────────────────
    fga_l10    = col("L10_FGA", 0.0)
    fg_pct_l10 = col("L10_FG_PCT", 0.45)
    fg3a_l10   = col("L10_FG3A", 0.0)
    fg3m_l10   = col("L10_FG3M", 0.0)
    fta_l10    = col("L10_FTA", 0.0)
    ft_rate    = col("L10_FT_PCT", 0.0)
    usage_l10  = col("L10_USAGE_APPROX", 0.18)
    usage_l30  = col("L30_USAGE_APPROX", 0.18)

    # ── Derived ──────────────────────────────────────────────────────────────
    pts_per_min    = float(L10 / max(min_l10, 1.0))
    fga_per_min    = float(fga_l10 / max(min_l10, 1.0))
    ppfga_l10      = float(L10 / max(fga_l10, 1.0))
    role_intensity = float(usage_l10 * pts_per_min)

    level        = L30
    reversion    = L10 - L30
    momentum     = L5  - L30
    acceleration = L3  - L5
    level_ewm    = float(pd.Series(pts_arr[-10:]).ewm(span=5, adjust=False).mean().iloc[-1]) if n >= 2 else L10
    consistency  = float(1.0 / (std10 + 1.0))
    volume       = float(L30 - line)
    trend        = float(L5  - L30)

    z_momentum  = momentum  / (std10 + 1e-6)
    z_reversion = reversion / (std10 + 1e-6)
    z_accel     = acceleration / (std10 + 1e-6)
    mean_rev    = abs(L10 - L30) / (std10 + 1e-6)
    extreme_hot  = float(L5 > L30 + std10)
    extreme_cold = float(L5 < L30 - std10)

    # Season progress (Oct 1 → Apr 20)
    from config import VERSION
    season_start = pd.Timestamp("2025-10-01")
    season_end   = pd.Timestamp("2026-04-20")
    sp = float(max(0.0, min(1.0, (game_date - season_start).days / max((season_end - season_start).days, 1))))
    games_depth    = min(float(n), 82.0)
    early_season_w = float(max(0.70, min(1.0, 0.70 + 0.30 * (games_depth / 30))))

    # Home/away
    is_home_arr = prior["IS_HOME"].fillna(True).astype(float).values if "IS_HOME" in prior.columns else np.ones(n)
    home_pts    = pts_arr[is_home_arr.astype(bool)]
    away_pts    = pts_arr[~is_home_arr.astype(bool)]
    home_l10    = _sm(home_pts[-10:], 10) if len(home_pts) > 0 else L10
    away_l10    = _sm(away_pts[-10:], 10) if len(away_pts) > 0 else L10
    home_away_split = home_l10 - away_l10

    # Rest
    is_b2b     = float(rest_days <= 1)
    rest_cat   = 0 if rest_days <= 1 else (1 if rest_days <= 2 else (2 if rest_days <= 4 else (3 if rest_days <= 6 else 4)))
    is_long    = float(rest_days >= 6)

    # Matchup
    pos_grp      = get_pos_group(pos_raw)
    defP_dynamic = float(dyn_dvp.get(f"{opponent}|{pos_grp}", 15))
    pace_r       = float(pace_rank.get(opponent, 15))
    opp_def_trend= float(opp_trend.get(f"{opponent}|{pos_grp}", 0.0))
    opp_def_var  = float(opp_var.get(f"{opponent}|{pos_grp}", 5.0))

    # Line context
    line_vs_l30   = float(line - L30)
    line_bucket   = float(min(int(line // 5), 5))
    line_spread   = float((max_line or line) - (min_line or line))
    line_sharpness= float(1.0 / (line_spread + 1.0))
    vol_risk      = float(line_spread * std10 / max(line, 1.0))

    # H2H
    h2h_ts_dev = h2h_fga_dev = h2h_min_dev = 0.0
    h2h_conf   = h2h_games   = h2h_trend   = 0.0
    h2h_avg    = None
    if h2h_row:
        def hf(k): return float(h2h_row.get(k, 0) or 0)
        h2h_ts_dev  = hf("H2H_TS_VS_OVERALL")
        h2h_fga_dev = hf("H2H_FGA_VS_OVERALL")
        h2h_min_dev = hf("H2H_MIN_VS_OVERALL")
        h2h_conf    = hf("H2H_CONFIDENCE")
        h2h_games   = hf("H2H_GAMES")
        h2h_trend   = hf("H2H_PTS_TREND")
        _a = h2h_row.get("H2H_AVG_PTS")
        try:
            h2h_avg = float(_a) if _a is not None else None
        except Exception:
            pass

    # Hit rates vs line
    l3_vs_line  = L3  - line
    l5_vs_line  = L5  - line
    l10_vs_line = L10 - line
    all_windows_over  = int(l3_vs_line > 0 and l5_vs_line > 0 and l10_vs_line > 0)
    all_windows_under = int(l3_vs_line < 0 and l5_vs_line < 0 and l10_vs_line < 0)

    # Long-term baseline (L100/L200 if available)
    long_term_pts = float(col("L100_PTS", col("L200_PTS", L30)))
    long_vs_short = long_term_pts - L10

    # Market features
    implied_over_prob  = None  # filled by caller from odds
    implied_under_prob = None
    books_log          = 0.0
    line_movement_norm = line_spread / max(L30, 1.0)

    return {
        # Rolling
        "L3": L3, "L5": L5, "L10": L10, "L20": L20, "L30": L30,
        "l3": L3, "l5": L5, "l10": L10, "l20": L20, "l30": L30,
        "std10": std10, "hr10": hr10, "hr30": hr30, "n_games": float(n),
        # Form
        "level": level, "reversion": reversion, "momentum": momentum,
        "acceleration": acceleration, "level_ewm": level_ewm,
        "z_momentum": z_momentum, "z_reversion": z_reversion, "z_accel": z_accel,
        "mean_reversion_risk": mean_rev, "extreme_hot": extreme_hot, "extreme_cold": extreme_cold,
        "consistency": consistency, "volume": volume, "trend": trend,
        "l3_vs_line": l3_vs_line, "l5_vs_line": l5_vs_line, "l10_vs_line": l10_vs_line,
        "all_windows_over": all_windows_over, "all_windows_under": all_windows_under,
        "long_term_pts": long_term_pts, "long_vs_short": long_vs_short,
        # Minutes/role
        "min_l10": min_l10, "min_l30": min_l30, "min_l3": min_l3,
        "minL10": min_l10, "minL30": min_l30,
        "min_cv": min_cv, "recent_min_trend": recent_min_trend,
        "pts_per_min": pts_per_min, "fga_per_min": fga_per_min,
        "ppfga_l10": ppfga_l10, "role_intensity": role_intensity,
        "minutes_cv_score": min_cv, "efficiency_l5": 0.0,
        # Shooting
        "fga_l10": fga_l10, "fg_pct_l10": fg_pct_l10,
        "fg3a_l10": fg3a_l10, "fg3m_l10": fg3m_l10,
        "fta_l10": fta_l10, "ft_rate": ft_rate, "ft_rate_l10": ft_rate,
        # Usage
        "usage_l10": usage_l10, "usage_l30": usage_l30,
        # Home/away
        "home_l10": home_l10, "away_l10": away_l10,
        "home_away_split": home_away_split,
        "is_home": float(1 if is_home else 0),
        # Rest
        "rest_days": float(rest_days), "is_b2b": is_b2b,
        "rest_cat": float(rest_cat), "is_long_rest": is_long,
        # Season
        "season_progress": sp, "early_season_weight": early_season_w,
        "games_depth": games_depth,
        # Matchup
        "defP_dynamic": defP_dynamic, "defP": defP_dynamic,
        "pace_rank": pace_r, "pace": pace_r,
        "opp_def_trend": opp_def_trend, "opp_def_var": opp_def_var,
        # Line
        "line": float(line), "line_vs_l30": line_vs_l30,
        "line_bucket": line_bucket, "line_spread": line_spread,
        "line_sharpness": line_sharpness, "vol_risk": vol_risk,
        "line_movement_norm": line_movement_norm,
        # H2H
        "h2h_ts_dev": h2h_ts_dev, "h2h_fga_dev": h2h_fga_dev,
        "h2h_min_dev": h2h_min_dev, "h2h_conf": h2h_conf,
        "h2h_games": h2h_games, "h2h_trend": h2h_trend,
        "h2h_avg": h2h_avg,
        "h2h_ts_delta": h2h_ts_dev, "h2h_fga_delta": h2h_fga_dev,
        "h2h_min_delta": h2h_min_dev,
        # Aliases
        "l10_ewm": level_ewm,
        "l5_ewm": float(pd.Series(pts_arr[-5:]).ewm(span=3, adjust=False).mean().iloc[-1]) if n >= 2 else L5,
        "pos_grp_str": pos_grp,
        "volatility": std10,
    }


# ── Recompute rolling after game log append ───────────────────────────────────

def recompute_rolling(df: pd.DataFrame, players: set[str]) -> pd.DataFrame:
    """
    Recompute pre-computed rolling columns for given players after appending
    new game log rows. Ensures batch_predict gets fresh L3/L5/L10/L30.
    """
    df = df.copy()
    df["GAME_DATE"] = pd.to_datetime(df["GAME_DATE"])

    def roll(arr: pd.Series, n: int) -> pd.Series:
        return arr.rolling(window=n, min_periods=1).mean()

    ROLL_COLS = [
        "L3_PTS", "L5_PTS", "L10_PTS", "L20_PTS", "L30_PTS",
        "L3_MIN_NUM", "L10_MIN_NUM", "L30_MIN_NUM",
        "L10_FGA", "L10_FG3A", "L10_FG3M", "L10_FTA",
        "L10_FT_PCT", "L10_USAGE_APPROX", "L30_USAGE_APPROX",
    ]

    for pname in players:
        mask = df["PLAYER_NAME"] == pname
        if not mask.any():
            continue
        grp = df[mask].sort_values("GAME_DATE").copy()
        idx = grp.index
        played_mask = (grp["MIN_NUM"].fillna(0) > 0) & (grp.get("DNP", pd.Series(0, index=grp.index)).fillna(0) == 0)

        pts_prior = grp["PTS"].where(played_mask).shift(1)
        grp["L3_PTS"]  = roll(pts_prior, 3)
        grp["L5_PTS"]  = roll(pts_prior, 5)
        grp["L10_PTS"] = roll(pts_prior, 10)
        grp["L20_PTS"] = roll(pts_prior, 20)
        grp["L30_PTS"] = roll(pts_prior, 30)

        min_prior = grp["MIN_NUM"].where(played_mask).shift(1)
        grp["L3_MIN_NUM"]  = roll(min_prior, 3)
        grp["L10_MIN_NUM"] = roll(min_prior, 10)
        grp["L30_MIN_NUM"] = roll(min_prior, 30)

        for src, dst, n in [
            ("FGA", "L10_FGA", 10), ("FG3A", "L10_FG3A", 10), ("FG3M", "L10_FG3M", 10),
            ("FTA", "L10_FTA", 10), ("FT_PCT", "L10_FT_PCT", 10),
            ("USAGE_APPROX", "L10_USAGE_APPROX", 10), ("USAGE_APPROX", "L30_USAGE_APPROX", 30),
        ]:
            if src in grp.columns:
                grp[dst] = roll(grp[src].where(played_mask).shift(1), n)

        cols = [c for c in ROLL_COLS if c in df.columns and c in grp.columns]
        if cols:
            df.loc[idx, cols] = grp[cols].values

    return df
