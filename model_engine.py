"""
PropEdge V2.0 — model_engine.py
────────────────────────────────────────────────────────────────────────────────
Loads the trained LightGBM model and isotonic calibrator.
Runs inference and assigns direction-specific confidence tiers.
Stateless — safe to call from any context.
────────────────────────────────────────────────────────────────────────────────
"""
from __future__ import annotations

import json
import numpy as np
import pandas as pd
import joblib

from config import MODEL_FILE, THRESHOLDS_FILE, TIER_THRESHOLDS
from feature_engine import MODEL_FEATURES, NEVER_USE_AS_FEATURES


# ── Model loader (cached in module scope) ─────────────────────────────────────

_model_cache: tuple | None = None


def load_model() -> tuple:
    """
    Load and cache (lgb_model, calibrator, feature_cols) from disk.
    Returns cached copy on subsequent calls.
    """
    global _model_cache
    if _model_cache is None:
        if not MODEL_FILE.exists():
            raise FileNotFoundError(
                f"Model not found: {MODEL_FILE}\n"
                f"Run: python3 run.py train"
            )
        _model_cache = joblib.load(MODEL_FILE)
    return _model_cache


def load_thresholds() -> dict[str, float]:
    """Load tier thresholds from JSON, falling back to config defaults."""
    if THRESHOLDS_FILE.exists():
        with open(THRESHOLDS_FILE) as f:
            return json.load(f)
    return TIER_THRESHOLDS.copy()


# ── Feature matrix builder ────────────────────────────────────────────────────

def build_X(df: pd.DataFrame, feature_cols: list[str]) -> np.ndarray:
    """
    Build model input matrix from DataFrame.
    Columns not present in df are filled with 0.
    Post-game columns are explicitly blocked.
    """
    # Safety: never let post-game columns slip into inference
    safe_cols = [c for c in feature_cols if c not in NEVER_USE_AS_FEATURES]
    X = df.reindex(columns=safe_cols, fill_value=0).fillna(0).values
    return X


# ── Tier assignment ───────────────────────────────────────────────────────────

def assign_tier(dir_conf: float, thresholds: dict[str, float]) -> str:
    """
    Assign tier label from direction-specific confidence.
    Uses cumulative thresholds (conf >= threshold → tier).
    """
    for tier in ["APEX", "ULTRA", "ELITE", "STRONG", "PLAY"]:
        if dir_conf >= thresholds[tier]:
            return tier
    return "SKIP"


# ── Main inference function ───────────────────────────────────────────────────

def predict(df: pd.DataFrame) -> pd.DataFrame:
    """
    Run full inference pipeline on a feature-engineered DataFrame.

    Returns df with added columns:
      p_over, p_under, direction, dir_conf, confidence_pct,
      tier, predicted_pts, market_edge_over, market_edge_under, value_flag
    """
    model, calibrator, feature_cols = load_model()
    thresholds = load_thresholds()

    X = build_X(df, feature_cols)

    # Raw probability from LightGBM
    raw_prob = model.predict_proba(X)[:, 1]

    # Isotonic calibration
    p_over = np.clip(calibrator.predict(raw_prob), 0.01, 0.99)
    p_under = 1.0 - p_over

    # Direction-specific confidence
    direction    = np.where(p_over >= 0.5, "OVER", "UNDER")
    dir_conf     = np.where(direction == "OVER", p_over, p_under)
    conf_pct     = np.round(dir_conf * 100, 1)

    # Tier
    tiers = np.array([assign_tier(c, thresholds) for c in dir_conf])

    # Predicted points
    pred_pts = np.where(
        direction == "OVER",
        p_over  * df["L10"].fillna(df.get("L30", 15)).values +
        p_under * df["L30"].fillna(15).values,
        p_under * df["L10"].fillna(df.get("L30", 15)).values +
        p_over  * df["L30"].fillna(15).values,
    )

    # Market edge
    imp_over  = df["implied_over_prob"].values  if "implied_over_prob"  in df.columns else 0.5
    imp_under = df["implied_under_prob"].values if "implied_under_prob" in df.columns else 0.5
    market_edge_over  = np.round(p_over  - imp_over,  4)
    market_edge_under = np.round(p_under - imp_under, 4)

    # Value flag from line_zscore
    lz = df["line_zscore"].values if "line_zscore" in df.columns else np.zeros(len(df))
    value_flag = np.where(
        lz <= -0.5,  "strong_over",
        np.where(lz <= 0, "mod_over",
        np.where(lz <= 0.5, "mod_under", "strong_under"))
    )

    out = df.copy()
    out["p_over"]            = np.round(p_over,  4)
    out["p_under"]           = np.round(p_under, 4)
    out["direction"]         = direction
    out["dir_conf"]          = np.round(dir_conf, 4)
    out["confidence_pct"]    = conf_pct
    out["tier"]              = tiers
    out["predicted_pts"]     = np.round(pred_pts, 1)
    out["market_edge_over"]  = market_edge_over
    out["market_edge_under"] = market_edge_under
    out["value_flag"]        = value_flag

    return out


# ── Scoreboard printer ────────────────────────────────────────────────────────

def print_scoreboard(acc_df: pd.DataFrame) -> None:
    """Print the standard V2.0 tier accuracy scoreboard."""
    print()
    print("╔══════════════════════════════════════════════════════════════════╗")
    print("║  PropEdge V2.0 — Tier Accuracy Report                          ║")
    print("╠══════════╦═══════╦══════════╦══════════╦═════════╦═════════╦════╣")
    print("║ Tier     ║   n   ║  OVER %  ║ UNDER %  ║ COMB %  ║ Target  ║ ✓  ║")
    print("╠══════════╬═══════╬══════════╬══════════╬═════════╬═════════╬════╣")
    for _, row in acc_df.iterrows():
        n   = int(row["n_total"])
        aov = row["acc_over"]   * 100
        aun = row["acc_under"]  * 100
        acb = row["acc_combined"] * 100
        tgt = row["acc_target"] * 100
        met = row["target_met"]
        gate = "✓" if met else ("⚠" if n < 30 else "✗")
        print(
            f"║ {row['tier']:<8} ║ {n:5d} ║ {aov:7.1f}%  ║ "
            f"{aun:7.1f}%  ║ {acb:6.1f}%  ║  {tgt:4.0f}%   ║ {gate}  ║"
        )
    print("╚══════════╩═══════╩══════════╩══════════╩═════════╩═════════╩════╝")
