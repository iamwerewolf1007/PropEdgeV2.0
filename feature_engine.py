"""
PropEdge V2.0 — feature_engine.py
────────────────────────────────────────────────────────────────────────────────
All feature computation as pure functions. No file I/O. No global state.
Takes DataFrames, returns DataFrames with new columns appended.

LEAKAGE CONTRACT:
  Post-game columns (actual_pts, actual_min, actual_fga, actual_fgm,
  actual_fta, actual_ftm, ts_pct, wl_win, plus_minus, result_over,
  result_under, is_push, direction_actual, fg_pct_l3, fg_pct_l5)
  are NEVER computed or included in model input features.

STREAK CONTRACT:
  hot_streak / cold_streak store the streak BEFORE the current game result
  is incorporated. The update runs AFTER the append — not before.
  This was the V9.x staleness bug. Fixed here permanently.
────────────────────────────────────────────────────────────────────────────────
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from config import H2H_MIN_GAMES, NEVER_USE_AS_FEATURES


# ── Safe converters ───────────────────────────────────────────────────────────

def _f(v, default: float = 0.0) -> float:
    try:
        return float(v) if v is not None and not (isinstance(v, float) and np.isnan(v)) else default
    except (TypeError, ValueError):
        return default


def _i(v, default: int = 0) -> int:
    try:
        return int(v) if v is not None else default
    except (TypeError, ValueError):
        return default


# ── American odds → implied probability ──────────────────────────────────────

def american_to_prob(odds: pd.Series) -> pd.Series:
    """Convert American odds to implied probability. Clips to [0.01, 0.99]."""
    odds = pd.to_numeric(odds, errors="coerce").fillna(-110)
    return np.where(
        odds < 0,
        -odds / (-odds + 100),
        100  / (odds + 100),
    ).clip(0.01, 0.99)


# ── Rolling stat features ─────────────────────────────────────────────────────

def add_rolling_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Rename game-log rolling columns to clean short names.
    All L{k}_PTS etc. are pre-game (confirmed: includes games up to but not
    including the current game — verified against LeBron Oct 28 2024 row).
    """
    rename = {}
    for window in [3, 5, 10, 20, 30, 50, 100, 200]:
        w = str(window)
        mapping = {
            f"L{w}_PTS":              f"L{window}",
            f"L{w}_MIN_NUM":          f"min_l{window}",
            f"L{w}_FGA":              f"fga_l{window}",
            f"L{w}_FTA":              f"fta_l{window}",
            f"L{w}_FG_PCT":           f"fg_pct_l{window}",
            f"L{w}_TRUE_SHOOTING_PCT":f"ts_l{window}",
            f"L{w}_USAGE_APPROX":     f"usage_l{window}",
            f"L{w}_PLUS_MINUS":       f"pm_l{window}",
            f"L{w}_WL_WIN":           f"wl_l{window}",
            f"L{w}_IS_HOME":          f"home_l{window}",
            f"L{w}_EFF_FG_PCT":       f"efg_l{window}",
            f"L{w}_FGM":              f"fgm_l{window}",
            f"L{w}_FTM":              f"ftm_l{window}",
        }
        rename.update({k: v for k, v in mapping.items() if k in df.columns})

    df = df.rename(columns=rename)

    # Fill nulls with position median then global median
    roll_cols = [f"L{w}" for w in [3,5,10,20,30,50]] + \
                ["min_l10","min_l30","usage_l10","usage_l30","fga_l10","fta_l10","fg_pct_l10"]
    for col in roll_cols:
        if col not in df.columns:
            df[col] = np.nan
        if "position" in df.columns:
            df[col] = df[col].fillna(df.groupby("position")[col].transform("median"))
        df[col] = df[col].fillna(df[col].median())

    return df


# ── std10 approximation ───────────────────────────────────────────────────────

def add_std10(df: pd.DataFrame) -> pd.DataFrame:
    """Approximate std10 from L10–L5 spread. Clipped to [1, 20]."""
    df["std10"] = ((df["L10"] - df["L5"]).abs() * 1.4).clip(1.0, 20.0)
    return df


# ── Market features ───────────────────────────────────────────────────────────

def add_market_features(df: pd.DataFrame) -> pd.DataFrame:
    df["implied_over_prob"]  = american_to_prob(df["over_odds"])
    df["implied_under_prob"] = american_to_prob(df["under_odds"])
    df["line_sharpness"]     = (df["min_line"] / df["max_line"].replace(0, np.nan)).fillna(1.0).clip(0, 1)
    df["books_log"]          = np.log1p(df["books"])
    df["line_spread"]        = df["max_line"] - df["min_line"]
    df["line_movement_norm"] = (df["line_spread"] / df["L30"].replace(0, 1)).abs().clip(0, 1)
    return df


# ── Form signals ──────────────────────────────────────────────────────────────

def add_form_features(df: pd.DataFrame) -> pd.DataFrame:
    df["line_zscore"]     = ((df["prop_line"] - df["L10"]) / df["std10"]).clip(-4, 4)
    df["volume"]          = df["L30"] - df["prop_line"]
    df["momentum"]        = df["L5"]  - df["L30"]
    df["accel"]           = df["L3"]  - df["L5"]
    df["reversion"]       = df["L10"] - df["L30"]
    df["l10_vs_line"]     = df["L10"] - df["prop_line"]
    df["l5_vs_line"]      = df["L5"]  - df["prop_line"]
    df["l3_vs_line"]      = df["L3"]  - df["prop_line"]
    df["all_windows_over"]  = (
        (df["L3"]  > df["prop_line"]) &
        (df["L5"]  > df["prop_line"]) &
        (df["L10"] > df["prop_line"])
    ).astype(int)
    df["all_windows_under"] = (
        (df["L3"]  < df["prop_line"]) &
        (df["L5"]  < df["prop_line"]) &
        (df["L10"] < df["prop_line"])
    ).astype(int)

    # Long-term baseline (100 or 200 game average)
    l100 = df["L100"] if "L100" in df.columns else df["L30"]
    l200 = df["L200"] if "L200" in df.columns else df["L30"]
    df["long_term_pts"] = l100.fillna(l200).fillna(df["L30"])
    df["long_vs_short"] = df["long_term_pts"] - df["L10"]

    return df


# ── Efficiency trend features ─────────────────────────────────────────────────

def add_trend_features(df: pd.DataFrame) -> pd.DataFrame:
    ts_l3  = df.get("ts_l3",  df["fg_pct_l10"])
    ts_l10 = df.get("ts_l10", df["fg_pct_l10"])
    fga_l3 = df.get("fga_l3", df["fga_l10"])
    fga_l30= df.get("fga_l30",df["fga_l10"])
    fg_l3  = df.get("fg_pct_l3", df["fg_pct_l10"]) if "fg_pct_l3" in df.columns else df["fg_pct_l10"]
    fg_l30 = df.get("fg_pct_l30",df["fg_pct_l10"])
    pm_l3  = df.get("pm_l3",  pd.Series(0.0, index=df.index))
    pm_l10 = df.get("pm_l10", pd.Series(0.0, index=df.index))
    use_l3 = df.get("usage_l3", df["usage_l10"])

    df["ts_trend_l3"]    = ts_l3.fillna(df["fg_pct_l10"])  - df["fg_pct_l10"]
    df["fg_trend"]       = fg_l3.fillna(df["fg_pct_l10"])  - fg_l30.fillna(df["fg_pct_l10"])
    df["fga_trend_l3"]   = fga_l3.fillna(df["fga_l10"])    - df["fga_l10"]
    df["usage_trend_l3"] = use_l3.fillna(df["usage_l10"])  - df["usage_l10"]
    df["pm_trend"]       = pm_l3.fillna(0) - pm_l10.fillna(0)
    df["efficiency_l5"]  = (
        (df["L5"] / df["min_l5"].replace(0, 1) if "min_l5" in df.columns
         else df["L5"] / df["min_l10"].replace(0, 1)) -
        (df["L30"] / df["min_l30"].replace(0, 1))
    ).clip(-2, 2)
    df["minutes_cv_score"] = (df["std10"] / df["L10"].replace(0, 1)).clip(0, 2)
    df["wl_l10_val"]     = df["wl_l10"].fillna(0.5) if "wl_l10" in df.columns else 0.5
    return df


# ── Role / position features ──────────────────────────────────────────────────

_POS_MAP = {
    "Guard": 0, "G": 0, "SG": 0, "PG": 0, "Guard-Forward": 0.5,
    "Forward": 1, "F": 1, "SF": 1, "PF": 1, "Center-Forward": 1.5,
    "Center": 2, "C": 2,
}

def add_role_features(df: pd.DataFrame) -> pd.DataFrame:
    df["pos_code"]    = df["position"].map(_POS_MAP).fillna(1.0)
    df["starter_flag"]= np.where(
        df["min_l10"] > 28, 1.0,
        np.where(df["min_l10"] > 20, 0.5, 0.0)
    )
    df["pos_line_pct"]= df.groupby(["position","season"])["prop_line"].rank(pct=True)
    return df


# ── Context features ──────────────────────────────────────────────────────────

def add_context_features(df: pd.DataFrame) -> pd.DataFrame:
    df["game_date"]    = pd.to_datetime(df["game_date"])
    df["day_of_week"]  = df["game_date"].dt.dayofweek

    s_min = df.groupby("season")["game_date"].transform("min")
    s_max = df.groupby("season")["game_date"].transform("max")
    df["season_pct"]   = ((df["game_date"] - s_min) / (s_max - s_min + pd.Timedelta("1D"))).clip(0, 1)
    df["season_pct_sq"]= df["season_pct"] ** 2
    df["season_segment"]= pd.cut(df["season_pct"], bins=5, labels=[0,1,2,3,4]).astype(float)

    df["post_allstar_flag"] = (
        ((df["game_date"].dt.month == 2) & (df["game_date"].dt.day > 18)) |
        (df["game_date"].dt.month >= 3)
    ).astype(int)

    return df


# ── H2H features ──────────────────────────────────────────────────────────────

def add_h2h_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Derive all H2H engineered features from the merged H2H database columns.
    Fallback to L30 for any missing H2H averages.
    """
    h2h_avg  = df["H2H_AVG_PTS"].fillna(df["L30"])
    h2h_l3   = df["L3_H2H_AVG_PTS"].fillna(df["L30"])
    h2h_home = df["H2H_HOME_AVG_PTS"].fillna(df["L30"])
    h2h_away = df["H2H_AWAY_AVG_PTS"].fillna(df["L30"])

    df["h2h_avg_pts"]        = h2h_avg
    df["h2h_l3_avg_pts"]     = h2h_l3
    df["h2h_games"]          = df["H2H_GAMES"].fillna(0)
    df["h2h_confidence"]     = df["H2H_CONFIDENCE"].fillna(0)
    df["h2h_predictability"] = df["H2H_PREDICTABILITY"].fillna(0.5)
    df["h2h_recency_weight"] = df["H2H_RECENCY_WEIGHT"].fillna(0.5)
    df["h2h_valid"]          = (df["h2h_games"] >= H2H_MIN_GAMES).astype(int)

    # Gap signals
    df["h2h_gap"]            = h2h_avg  - df["prop_line"]
    df["h2h_l3_gap"]         = h2h_l3   - df["prop_line"]
    df["h2h_gap_abs"]        = df["h2h_gap"].abs()
    df["h2h_home_gap"]       = h2h_home - df["prop_line"]
    df["h2h_away_gap"]       = h2h_away - df["prop_line"]
    df["h2h_home_away_edge"] = np.where(
        df.get("is_home", 0) == 1,
        df["h2h_home_gap"],
        df["h2h_away_gap"],
    )
    df["h2h_win_loss_gap"]   = (
        df["H2H_WIN_AVG_PTS"].fillna(0) - df["H2H_LOSS_AVG_PTS"].fillna(0)
    )
    df["h2h_season_drift"]   = df["H2H_SEASON_DRIFT"].fillna(0)
    df["h2h_pts_trend_val"]  = df["H2H_PTS_TREND"].fillna(0)
    df["h2h_ha_diff"]        = df["H2H_HOME_AWAY_DIFF"].fillna(0)
    df["h2h_ts_delta"]       = df["H2H_TS_VS_OVERALL"].fillna(0)
    df["h2h_fga_delta"]      = df["H2H_FGA_VS_OVERALL"].fillna(0)
    df["h2h_min_delta"]      = df["H2H_MIN_VS_OVERALL"].fillna(0)

    # Consensus flags
    df["h2h_l10_agree"]  = (
        np.sign(df["h2h_gap"].fillna(0)) == np.sign(df["l10_vs_line"])
    ).astype(int)
    df["h2h_l5_agree"]   = (
        np.sign(df["h2h_gap"].fillna(0)) == np.sign(df["l5_vs_line"])
    ).astype(int)
    df["h2h_all_agree"]  = (
        (df["h2h_gap"].fillna(0) > 0) &
        (df["l10_vs_line"] > 0) &
        (df["l5_vs_line"]  > 0)
    ).astype(int)
    df["h2h_all_under"]  = (
        (df["h2h_gap"].fillna(0) < 0) &
        (df["l10_vs_line"] < 0) &
        (df["l5_vs_line"]  < 0)
    ).astype(int)

    return df


# ── Player signals (streaks, personal rates, recency HR) ─────────────────────

def add_player_signals(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute per-player, chronologically ordered signals.
    All values at row i reflect only games BEFORE game i — no leakage.

    STREAK CONTRACT:
      hot.append(h)  ← appended BEFORE updating with current result
      if r == 1: h += 1 ...  ← update happens AFTER append
    """
    df = df.sort_values(["player_name", "game_date"]).reset_index(drop=True)

    # actual_pts only exists for graded historical rows — live prediction rows have no result yet.
    # Fill with 0.5 (neutral) so streak/rate computation starts clean for live rows.
    if "actual_pts" in df.columns:
        _pts = df["actual_pts"].fillna(df["prop_line"])
    else:
        _pts = df["prop_line"]  # no result available — neutral baseline
    result = (_pts > df["prop_line"]).astype(float)

    hot_list, cold_list   = [], []
    per_ov_list, per_un_list = [], []
    rw_ov_list, rw_un_list   = [], []

    for player, grp in df.groupby("player_name", sort=False):
        idx      = grp.index.tolist()
        results  = result.iloc[idx].values
        lines    = grp["prop_line"].values
        h = c = 0
        bkt_ov: dict[float, list] = {}
        bkt_un: dict[float, list] = {}

        for i, (r, line) in enumerate(zip(results, lines)):
            bkt = round(line / 2.5) * 2.5

            # Personal rates: history BEFORE this game
            per_ov_list.append(np.mean(bkt_ov[bkt]) if bkt in bkt_ov and bkt_ov[bkt] else 0.5)
            per_un_list.append(np.mean(bkt_un[bkt]) if bkt in bkt_un and bkt_un[bkt] else 0.5)

            # Recency-weighted HR: up to 10 prior games, L3×3, L4-7×2, L8-10×1
            hist = results[max(0, i - 10):i]
            if len(hist) == 0:
                rw_ov_list.append(0.5)
                rw_un_list.append(0.5)
            else:
                wts = [(v, 3 if j < 3 else (2 if j < 7 else 1))
                       for j, v in enumerate(reversed(hist))]
                tw  = sum(w for _, w in wts)
                ov_ = sum(v * w for v, w in wts) / tw
                rw_ov_list.append(ov_)
                rw_un_list.append(1 - ov_)

            # ── STREAK CONTRACT: append PRE-game value, then update ───────────
            hot_list.append(h)
            cold_list.append(c)
            if r == 1:
                h += 1; c = 0
            else:
                c += 1; h = 0
            # ─────────────────────────────────────────────────────────────────

            bkt_ov.setdefault(bkt, []).append(r)
            bkt_un.setdefault(bkt, []).append(1 - r)

    df["hot_streak"]             = hot_list
    df["cold_streak"]            = cold_list
    df["personal_over_rate"]     = per_ov_list
    df["personal_under_rate"]    = per_un_list
    df["recency_weighted_hr_over"]  = rw_ov_list
    df["recency_weighted_hr_under"] = rw_un_list

    return df


# ── Full feature pipeline ─────────────────────────────────────────────────────

def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply all feature engineering steps in order.
    Input: merged DataFrame (lines + game logs + H2H joined).
    Output: same DataFrame with all engineered features appended.
    """
    df = add_rolling_features(df)
    df = add_std10(df)
    df = add_market_features(df)
    df = add_form_features(df)
    df = add_trend_features(df)
    df = add_role_features(df)
    df = add_context_features(df)
    df = add_h2h_features(df)
    df = add_player_signals(df)
    return df


# ── Feature column list (safe, pre-game only) ─────────────────────────────────

MODEL_FEATURES: list[str] = [
    # Rolling averages
    "L3", "L5", "L10", "L20", "L30", "L50", "L100", "L200", "std10",
    # Form
    "line_zscore", "volume", "momentum", "accel", "reversion",
    "recency_weighted_hr_over", "recency_weighted_hr_under",
    "personal_over_rate", "personal_under_rate",
    "hot_streak", "cold_streak",
    # H2H
    "H2H_CONFIDENCE", "H2H_PREDICTABILITY", "H2H_RECENCY_WEIGHT",
    "H2H_GAMES", "H2H_AVG_PTS", "H2H_PTS_TREND", "H2H_SEASON_DRIFT",
    "H2H_HOME_AVG_PTS", "H2H_AWAY_AVG_PTS", "H2H_HOME_AWAY_DIFF",
    "H2H_WIN_AVG_PTS", "H2H_LOSS_AVG_PTS", "H2H_STD_PTS",
    "H2H_TS_VS_OVERALL", "H2H_FGA_VS_OVERALL", "H2H_MIN_VS_OVERALL",
    "H2H_USAGE_VS_OVERALL", "H2H_CURRENT_SEASON_AVG_PTS",
    "L3_H2H_AVG_PTS", "L5_H2H_AVG_PTS",
    "h2h_gap", "h2h_l3_gap", "h2h_gap_abs", "h2h_home_away_edge",
    "h2h_win_loss_gap", "h2h_season_drift", "h2h_pts_trend_val", "h2h_ha_diff",
    "h2h_games", "h2h_confidence", "h2h_predictability",
    "h2h_valid", "h2h_avg_pts", "h2h_l3_avg_pts",
    "h2h_l10_agree", "h2h_l5_agree", "h2h_all_agree", "h2h_all_under",
    "h2h_away_gap", "h2h_home_gap",
    # Direction composites
    "all_windows_over", "all_windows_under",
    "l10_vs_line", "l5_vs_line", "l3_vs_line",
    "long_term_pts", "long_vs_short",
    # Minutes / role
    "starter_flag", "min_l10", "min_l30", "minutes_cv_score", "efficiency_l5",
    # Usage
    "usage_l10", "usage_l30",
    # Shooting trends
    "ts_l3", "ts_l5", "ts_l10", "efg_l3", "efg_l5", "efg_l10",
    "fg_pct_l10", "fg_pct_l30", "ts_trend_l3", "fg_trend",
    "fga_l10", "fga_l30", "fga_trend_l3",
    "fta_l10", "usage_trend_l3",
    # Plus/minus
    "pm_l3", "pm_l10", "pm_l30", "pm_trend",
    # Win rate / home
    "wl_l3", "wl_l5", "wl_l10_val", "home_l3", "home_l5",
    # Context
    "opp_b2b", "is_home", "season_pct", "season_pct_sq",
    "post_allstar_flag", "pos_code", "day_of_week",
    "season_segment", "pos_line_pct",
    # Market
    "line_sharpness", "line_spread", "line_movement_norm",
    "implied_over_prob", "implied_under_prob", "books_log", "prop_line",
]

# Verified no post-game columns in list
assert not (set(MODEL_FEATURES) & NEVER_USE_AS_FEATURES), \
    "LEAK DETECTED: post-game column in MODEL_FEATURES"
