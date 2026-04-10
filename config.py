"""
PropEdge V2.0 — config.py
Single source of truth for all paths, constants, thresholds, and helpers.
Every other module imports from here — never hardcode values elsewhere.
"""
from __future__ import annotations

import json
import math
from datetime import datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

VERSION     = "V2.0"
VERSION_TAG = "PropEdge V2.0"

ROOT        = Path(__file__).parent.resolve()
DATA_DIR    = ROOT / "data"
MODELS_DIR  = ROOT / "models"
OUTPUT_DIR  = ROOT / "output"
LOGS_DIR    = ROOT / "logs"
SOURCE_DIR  = ROOT / "source-files"

for _d in (DATA_DIR, MODELS_DIR, OUTPUT_DIR, LOGS_DIR, SOURCE_DIR,
           DATA_DIR / "monthly" / "2024_25",
           DATA_DIR / "monthly" / "2025_26"):
    _d.mkdir(parents=True, exist_ok=True)

# ── Source files (place in source-files/) ─────────────────────────────────────
FILE_GL_2425   = SOURCE_DIR / "gamelogs_2024_25.csv"
FILE_GL_2526   = SOURCE_DIR / "gamelogs_2025_26.csv"
FILE_H2H       = SOURCE_DIR / "h2h_database.csv"
FILE_PROPS_2425 = SOURCE_DIR / "Player_lines_2024-25.xlsx"
FILE_PROPS_2526 = SOURCE_DIR / "Player_lines_2025-26.xlsx"

# ── Data output files ─────────────────────────────────────────────────────────
FILE_TODAY        = DATA_DIR / "today.json"
FILE_SEASON_2425  = DATA_DIR / "season_2024_25.json"
FILE_SEASON_2526  = DATA_DIR / "season_2025_26.json"
FILE_DVP          = DATA_DIR / "dvp_rankings.json"

# ── Model files ───────────────────────────────────────────────────────────────
MODEL_FILE      = MODELS_DIR / "v2_lgbm.pkl"
THRESHOLDS_FILE = MODELS_DIR / "thresholds.json"

# ── Output artefacts ──────────────────────────────────────────────────────────
FEATURES_CSV      = OUTPUT_DIR / "propedge_features.csv"
DATASET_CSV       = OUTPUT_DIR / "propedge_v2_ml_dataset.csv"
DATASET_XLSX      = OUTPUT_DIR / "propedge_v2_ml_dataset.xlsx"
SCHEMA_CSV        = OUTPUT_DIR / "propedge_v2_schema.csv"
TIER_ACCURACY_CSV = OUTPUT_DIR / "tier_accuracy.csv"
CEILING_TXT       = OUTPUT_DIR / "ceiling_analysis.txt"

# ── Tier thresholds (direction-specific confidence = max(p_over, p_under)) ────
TIER_THRESHOLDS: dict[str, float] = {
    "APEX":   0.98,
    "ULTRA":  0.93,
    "ELITE":  0.88,
    "STRONG": 0.78,
    "PLAY":   0.65,
}

TIER_TARGETS: dict[str, float] = {
    "APEX":    0.98,
    "ULTRA":   0.98,
    "ELITE":   0.85,
    "STRONG":  0.85,
    "PLAY":    0.85,
    "OVERALL": 0.75,
}

TIER_STAKES: dict[str, float] = {
    "APEX": 3.0, "ULTRA": 2.0, "ELITE": 1.5,
    "STRONG": 1.0, "PLAY": 0.5, "SKIP": 0.0,
}

MIN_SAMPLE_GATE = 30
H2H_MIN_GAMES   = 5
MIN_PRIOR_GAMES  = 5

# ── Post-game leakage columns — NEVER model inputs ────────────────────────────
NEVER_USE_AS_FEATURES = {
    "actual_pts", "actual_min", "actual_fga", "actual_fgm",
    "actual_fta", "actual_ftm", "usage_actual", "ts_pct",
    "wl_win", "plus_minus", "result_over", "result_under",
    "is_push", "direction_actual", "fg_pct_l3", "fg_pct_l5",
    "hot_streak_length", "cold_streak_length",
}

# ── Odds API ──────────────────────────────────────────────────────────────────
ODDS_API_KEY  = "a77b14b513399a472139e58390aac514"
ODDS_API_BASE = "https://api.the-odds-api.com/v4"
ODDS_SPORT    = "basketball_nba"
ODDS_MARKET   = "player_points"
CREDIT_ALERT  = 50

# ── GitHub ────────────────────────────────────────────────────────────────────
GITHUB_OWNER  = "iamwerewolf1007"
GITHUB_REPO   = "PropEdgeV2.0"
GITHUB_BRANCH = "main"
GIT_REMOTE    = "git@github.com:iamwerewolf1007/PropEdgeV2.0.git"
LOCAL_DIR     = Path.home() / "Documents" / "GitHub" / "PropEdgeV2.0Local"
REPO_DIR      = Path.home() / "Documents" / "GitHub" / "PropEdgeV2.0"

PUSH_FILES_PREDICT = ["data/today.json", "data/dvp_rankings.json"]
PUSH_FILES_GRADE   = ["data/today.json", "data/dvp_rankings.json"]

# ── LightGBM training params ──────────────────────────────────────────────────
WF_MIN_TRAIN_ROWS = 500
WF_MIN_TEST_ROWS  = 15
WF_VAL_FRACTION   = 0.12
WF_MIN_VAL_ROWS   = 50

LGB_PARAMS: dict = {
    "n_estimators":      1000,
    "learning_rate":     0.02,
    "max_depth":         7,
    "num_leaves":        127,
    "min_child_samples": 15,
    "subsample":         0.8,
    "colsample_bytree":  0.75,
    "reg_alpha":         0.3,
    "reg_lambda":        1.0,
    "random_state":      42,
    "n_jobs":            -1,
    "verbosity":         -1,
}
LGB_EARLY_STOPPING = 60

# ── Position mapping ──────────────────────────────────────────────────────────
POS_MAP = {
    "PG": "Guard", "SG": "Guard", "G": "Guard", "GF": "Guard", "FG": "Guard",
    "SF": "Forward", "PF": "Forward", "F": "Forward", "FC": "Forward", "CF": "Forward",
    "C": "Center",
}

def get_pos_group(pos: str) -> str:
    p = str(pos).strip().upper()
    return POS_MAP.get(p, POS_MAP.get(p.split("-")[0], "Guard"))

# ── Timezone helpers ──────────────────────────────────────────────────────────
_ET = ZoneInfo("America/New_York")
_UK = ZoneInfo("Europe/London")

def now_et() -> datetime:
    return datetime.now(_ET)

def now_uk() -> datetime:
    return datetime.now(_UK)

def today_et() -> str:
    return datetime.now(_ET).strftime("%Y-%m-%d")

def et_window(date_str: str) -> tuple[str, str]:
    """Return UTC window for a given ET date."""
    y, m, d = int(date_str[:4]), int(date_str[5:7]), int(date_str[8:10])
    midnight_et = datetime(y, m, d, 0, 0, tzinfo=_ET)
    frm = (midnight_et - timedelta(hours=6)).astimezone(timezone.utc)
    to  = (midnight_et + timedelta(hours=30)).astimezone(timezone.utc)
    return frm.strftime("%Y-%m-%dT%H:%M:%SZ"), to.strftime("%Y-%m-%dT%H:%M:%SZ")

def current_season(dt: datetime | None = None) -> str:
    if dt is None:
        dt = datetime.now(_ET)
    yr, mo = dt.year, dt.month
    return f"{yr}-{str(yr+1)[-2:]}" if mo >= 10 else f"{yr-1}-{str(yr)[-2:]}"

# ── Tier assignment ───────────────────────────────────────────────────────────
def assign_tier(dir_conf: float, thresholds: dict[str, float] | None = None) -> str:
    thr = thresholds or TIER_THRESHOLDS
    for tier in ["APEX", "ULTRA", "ELITE", "STRONG", "PLAY"]:
        if dir_conf >= thr[tier]:
            return tier
    return "SKIP"

# ── JSON serialiser (handles numpy types, NaN, Inf) ───────────────────────────
def clean_json(obj):
    try:
        import numpy as np
        if isinstance(obj, dict):
            return {k: clean_json(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [clean_json(v) for v in obj]
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            v = float(obj)
            return None if (math.isnan(v) or math.isinf(v)) else v
        if isinstance(obj, np.bool_):
            return bool(obj)
        if isinstance(obj, float):
            return None if (math.isnan(obj) or math.isinf(obj)) else obj
        return obj
    except ImportError:
        if isinstance(obj, dict):
            return {k: clean_json(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [clean_json(v) for v in obj]
        if isinstance(obj, float):
            return None if (math.isnan(obj) or math.isinf(obj)) else obj
        return obj
