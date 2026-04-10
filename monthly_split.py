"""
PropEdge V2.0 — monthly_split.py
Splits season JSON files into per-month files for GitHub Pages hosting.

FILE LAYOUT:
  data/monthly/2025_26/index.json
  data/monthly/2025_26/2025-10.json
  data/monthly/2024_25/index.json
  data/monthly/2024_25/2024-10.json

Atomic writes — .tmp → rename. One .bak kept per file.
"""
from __future__ import annotations

import json
import os
import shutil
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

ROOT     = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data" / "monthly"


def _monthly_dir(season_key: str) -> Path:
    return DATA_DIR / season_key

def _month_path(season_key: str, month_str: str) -> Path:
    return _monthly_dir(season_key) / f"{month_str}.json"

def _index_path(season_key: str) -> Path:
    return _monthly_dir(season_key) / "index.json"

def _atomic_write(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    try:
        tmp.write_text(json.dumps(data, separators=(",", ":"), ensure_ascii=False))
        if path.exists():
            shutil.copy2(path, path.with_suffix(".bak"))
        os.replace(tmp, path)
    except Exception:
        if tmp.exists():
            tmp.unlink(missing_ok=True)
        raise

def _group_by_month(plays: list[dict]) -> dict[str, list[dict]]:
    groups: dict[str, list[dict]] = defaultdict(list)
    for p in plays:
        month = str(p.get("date", ""))[:7]
        if month and len(month) == 7:
            groups[month].append(p)
    return dict(groups)


def write_monthly_split(plays: list[dict], season_key: str) -> dict[str, int]:
    """Write all monthly files. Returns {month: count}."""
    by_month = _group_by_month(plays)
    counts: dict[str, int] = {}
    for month in sorted(by_month):
        month_plays = sorted(by_month[month],
                             key=lambda p: (p.get("date", ""), p.get("player", "")))
        _atomic_write(_month_path(season_key, month), month_plays)
        counts[month] = len(month_plays)

    index = {
        "season":      season_key,
        "months":      sorted(counts.keys()),
        "counts":      counts,
        "total_plays": sum(counts.values()),
        "updated_at":  datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }
    _atomic_write(_index_path(season_key), index)

    total = sum(counts.values())
    if total != len(plays):
        raise ValueError(f"Monthly split mismatch: {total} != {len(plays)}")
    return counts


def update_month(plays_for_month: list[dict], season_key: str, month_str: str) -> None:
    """Update a single month file after daily grading."""
    path = _month_path(season_key, month_str)
    existing: list[dict] = []
    if path.exists():
        try:
            existing = json.loads(path.read_text())
        except Exception:
            existing = []

    by_key = {(p.get("player", ""), p.get("date", "")): p for p in existing}
    for p in plays_for_month:
        by_key[(p.get("player", ""), p.get("date", ""))] = p

    merged = sorted(by_key.values(), key=lambda p: (p.get("date", ""), p.get("player", "")))
    _atomic_write(path, merged)
    _refresh_index(season_key)


def _refresh_index(season_key: str) -> None:
    monthly_dir = _monthly_dir(season_key)
    if not monthly_dir.exists():
        return
    counts: dict[str, int] = {}
    for f in sorted(monthly_dir.glob("????-??.json")):
        try:
            counts[f.stem] = len(json.loads(f.read_text()))
        except Exception:
            pass
    index = {
        "season":      season_key,
        "months":      sorted(counts.keys()),
        "counts":      counts,
        "total_plays": sum(counts.values()),
        "updated_at":  datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }
    _atomic_write(_index_path(season_key), index)


def load_monthly_split(season_key: str) -> list[dict]:
    monthly_dir = _monthly_dir(season_key)
    if not monthly_dir.exists():
        return []
    plays: list[dict] = []
    for f in sorted(monthly_dir.glob("????-??.json")):
        try:
            plays.extend(json.loads(f.read_text()))
        except Exception:
            pass
    return plays


def get_push_paths(season_key: str, only_current_month: bool = False) -> list[str]:
    from datetime import date
    if only_current_month:
        current = date.today().strftime("%Y-%m")
        return [
            f"data/monthly/{season_key}/{current}.json",
            f"data/monthly/{season_key}/index.json",
        ]
    monthly_dir = _monthly_dir(season_key)
    if not monthly_dir.exists():
        return []
    return [f.relative_to(ROOT).as_posix() for f in sorted(monthly_dir.glob("*.json"))]


def verify_monthly_integrity(season_key: str, full_plays: list[dict]) -> tuple[bool, str]:
    path = _index_path(season_key)
    if not path.exists():
        return False, f"No index for {season_key}"
    try:
        index = json.loads(path.read_text())
    except Exception as e:
        return False, str(e)
    total = index.get("total_plays", 0)
    if total != len(full_plays):
        return False, f"{season_key}: {total} != {len(full_plays)}"
    return True, f"{season_key}: {len(index['months'])} months, {total:,} plays verified"
