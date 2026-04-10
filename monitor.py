"""
PropEdge V2.0 — monitor.py
────────────────────────────────────────────────────────────────────────────────
Rolling 30-day accuracy, calibration drift, and tier health monitoring.
Reads from season JSON files — no retraining required.

RUN: python3 run.py monitor
     python3 run.py monitor --days 60
────────────────────────────────────────────────────────────────────────────────
"""
from __future__ import annotations

import json
import warnings
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from config import (
    SEASON_2425_JSON, SEASON_2526_JSON,
    TIER_TARGETS, MIN_SAMPLE_GATE, VERSION_TAG,
)

warnings.filterwarnings("ignore")


def load_all_graded(window_days: int = 30) -> list[dict]:
    """Load graded plays from both season JSONs within the rolling window."""
    cutoff = (date.today() - timedelta(days=window_days)).isoformat()
    plays  = []
    for path in [SEASON_2425_JSON, SEASON_2526_JSON]:
        if not path.exists():
            continue
        with open(path) as f:
            data = json.load(f)
        for play in data.get("plays", []):
            if (play.get("result") in ("WIN","LOSS") and
                    play.get("game_date", "9999") >= cutoff):
                plays.append(play)
    return plays


def compute_ece(plays: list[dict]) -> float:
    """Expected calibration error from confidence vs accuracy."""
    try:
        from sklearn.calibration import calibration_curve
        confs = [p["confidence_pct"] / 100 for p in plays if "confidence_pct" in p]
        labels= [1 if p["result"]=="WIN" else 0 for p in plays if "confidence_pct" in p]
        if len(confs) < 30:
            return float("nan")
        frac, pred_m = calibration_curve(labels, confs, n_bins=8)
        return float(np.mean(np.abs(frac - pred_m)))
    except Exception:
        return float("nan")


def run_monitor(window_days: int = 30) -> None:
    today_str = date.today().strftime("%Y-%m-%d")
    print(f"\n  {VERSION_TAG} — Monitor  [{today_str}]")
    print(f"  Rolling window: {window_days} days")
    print("  " + "─" * 52)

    plays = load_all_graded(window_days)
    if not plays:
        print(f"  ⚠ No graded plays in last {window_days} days.")
        print("    Run: python3 run.py grade  after games complete.")
        return

    print(f"  Graded plays in window: {len(plays)}")

    alerts: list[str] = []

    # Per-tier accuracy
    print()
    print(f"  {'Tier':<10} {'n':>6}  {'Acc':>8}  {'Target':>8}  Status")
    print("  " + "─" * 48)
    for tier, tgt in TIER_TARGETS.items():
        subset = [p for p in plays if (p.get("tier")==tier if tier!="OVERALL"
                  else p.get("tier")!="SKIP")]
        n = len(subset)
        if n < 10:
            print(f"  {tier:<10} {'INSUFF':>6}  {'—':>8}  {tgt*100:>6.0f}%   ⚠")
            continue
        acc = sum(1 for p in subset if p["result"]=="WIN") / n
        gate = "✓" if acc >= tgt else "✗ ALERT"
        print(f"  {tier:<10} {n:>6}  {acc*100:>7.1f}%  {tgt*100:>6.0f}%   {gate}")
        if acc < tgt and n >= MIN_SAMPLE_GATE:
            alerts.append(f"{tier}: {acc*100:.1f}% < {tgt*100:.0f}% target (n={n})")

    # ECE calibration
    ece = compute_ece(plays)
    if not np.isnan(ece):
        flag = "✓" if ece < 0.03 else "✗ RECALIBRATE"
        print(f"\n  Calibration ECE: {ece:.4f}  {flag}")
        if ece >= 0.03:
            alerts.append(f"ECE={ece:.4f} >= 0.03 — recalibration recommended")

    # Direction split
    over_plays  = [p for p in plays if p.get("direction")=="OVER"]
    under_plays = [p for p in plays if p.get("direction")=="UNDER"]
    if over_plays:
        acc_ov = sum(1 for p in over_plays  if p["result"]=="WIN") / len(over_plays)
        print(f"\n  OVER  calls:  {acc_ov*100:.1f}% ({len(over_plays)} plays)")
    if under_plays:
        acc_un = sum(1 for p in under_plays if p["result"]=="WIN") / len(under_plays)
        print(f"  UNDER calls:  {acc_un*100:.1f}% ({len(under_plays)} plays)")

    # Season breakdown
    seasons = set(p.get("season","") for p in plays)
    if len(seasons) > 1:
        print()
        for s in sorted(seasons):
            sp = [p for p in plays if p.get("season")==s]
            if sp:
                acc = sum(1 for p in sp if p["result"]=="WIN") / len(sp)
                print(f"  {s}: {acc*100:.1f}% (n={len(sp)})")

    # Alerts
    print()
    if alerts:
        print(f"  ⚠ {len(alerts)} alert(s):")
        for a in alerts:
            print(f"    — {a}")
    else:
        print("  ✓ All systems nominal — no alerts.")

    print()
