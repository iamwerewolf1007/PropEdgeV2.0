"""
PropEdge V2.0 — run.py
Master CLI orchestrator.

Commands:
  python3 run.py generate                     Full rebuild: both season JSONs + monthly split
  python3 run.py grade                        B0: grade yesterday + append game log
  python3 run.py grade --date 2026-04-05     Grade specific date
  python3 run.py predict                      B1: fetch props + predict today
  python3 run.py fetch-and-predict            Fetch from Odds API + predict (full daily workflow)
  python3 run.py fetch-and-predict --no-push
  python3 run.py train                        Walk-forward train + save model
  python3 run.py train --mode validate        OOS evaluation only
  python3 run.py train --fast                 20% sample
  python3 run.py generate-dataset            Build ML dataset CSV + XLSX + schema
  python3 run.py monitor                      Rolling accuracy + alerts
  python3 run.py monitor --days 60
  python3 run.py token-check                  Diagnose GitHub token
  python3 run.py name-test                    Player name alias self-test (40 cases)
  python3 run.py check                        Data integrity check

Daily workflow:
  python3 run.py fetch-and-predict            Morning: fetch props + predict
  python3 run.py grade                        After games: grade + update game log
  python3 run.py monitor                      Weekly: check accuracy

First-time setup:
  Place in source-files/:
    gamelogs_2024_25.csv
    gamelogs_2025_26.csv
    h2h_database.csv
    Player_lines_2024-25.xlsx
    Player_lines_2025-26.xlsx
  python3 run.py generate          (builds season JSONs, ~15 min)
  python3 run.py train             (trains V2.0 model)
  python3 run.py fetch-and-predict (daily from here)
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).parent.resolve()
sys.path.insert(0, str(ROOT))

# Clear stale bytecode
import shutil as _sh
_cache = ROOT / "__pycache__"
if _cache.exists():
    try: _sh.rmtree(_cache); _cache.mkdir(exist_ok=True)
    except Exception: pass

from config import VERSION_TAG, today_et


def _run(script: str, *args: str) -> int:
    cmd = [sys.executable, str(ROOT / script)] + list(args)
    return subprocess.run(cmd, cwd=ROOT).returncode


# ─────────────────────────────────────────────────────────────────────────────

def cmd_generate() -> None:
    print(f"\n  {VERSION_TAG} — Generate")
    from config import FILE_GL_2425, FILE_GL_2526, FILE_PROPS_2425, FILE_PROPS_2526, FILE_H2H
    missing = [f for f in (FILE_GL_2425, FILE_GL_2526, FILE_H2H) if not f.exists()]
    if missing:
        print("  ✗ Missing source files:")
        for f in missing: print(f"    {f}")
        return
    _run("generate_season_json.py")


def cmd_grade(date_str: str | None = None, push: bool = True) -> None:
    from batch_grade import run_grade
    run_grade(date_str=date_str, push=push)


def cmd_predict(push: bool = True) -> None:
    from batch_predict import run_predict
    run_predict(push=push)


def cmd_fetch_and_predict(date_str: str | None = None, push: bool = True) -> None:
    from fetch_props import fetch_props_for_predict
    from batch_predict import run_predict
    from config import SOURCE_DIR
    games = fetch_props_for_predict(date_str=date_str)
    if not games:
        print("  ✗ No props fetched.")
        return
    run_predict(date_str=date_str, push=push, games=games)


def cmd_train(mode: str = "train", fast: bool = False) -> None:
    from train import main as train_main
    train_main(mode=mode, fast=fast)


def cmd_generate_dataset() -> None:
    from generate_dataset import generate_dataset
    generate_dataset()


def cmd_monitor(days: int = 30) -> None:
    from monitor import run_monitor
    run_monitor(window_days=days)


def cmd_token_check() -> None:
    from git_push import token_check
    token_check()


def cmd_name_test() -> None:
    result = subprocess.run([sys.executable, str(ROOT / "player_name_aliases.py")], cwd=ROOT)
    sys.exit(result.returncode)


def cmd_check() -> None:
    print(f"\n  {VERSION_TAG} — Data Integrity Check")
    from config import (FILE_GL_2425, FILE_GL_2526, FILE_H2H, FILE_PROPS_2425, FILE_PROPS_2526,
                        FILE_SEASON_2425, FILE_SEASON_2526, MODEL_FILE, THRESHOLDS_FILE)
    import json
    checks = {
        "Game log 2024-25":     FILE_GL_2425,
        "Game log 2025-26":     FILE_GL_2526,
        "H2H database":         FILE_H2H,
        "Props 2024-25 Excel":  FILE_PROPS_2425,
        "Props 2025-26 Excel":  FILE_PROPS_2526,
        "Season 2024-25 JSON":  FILE_SEASON_2425,
        "Season 2025-26 JSON":  FILE_SEASON_2526,
        "V2.0 model pkl":       MODEL_FILE,
        "Thresholds JSON":      THRESHOLDS_FILE,
    }
    all_ok = True
    for label, path in checks.items():
        exists = path.exists()
        size   = f"{path.stat().st_size/1024:.0f} KB" if exists else "—"
        sym    = "✓" if exists else "✗"
        if not exists: all_ok = False
        print(f"  {sym} {label:<26} {size:>10}   {path.name}")

    for sf, label in ((FILE_SEASON_2526, "2025-26"), (FILE_SEASON_2425, "2024-25")):
        if sf.exists():
            try:
                plays = json.loads(sf.read_text())
                graded = [p for p in plays if p.get("result") in ("WIN","LOSS")]
                wins   = sum(1 for p in graded if p.get("result")=="WIN")
                hr     = f"{wins/len(graded)*100:.1f}%" if graded else "—"
                print(f"  Season {label}: {len(plays):,} plays | {len(graded):,} graded | HR={hr}")
            except Exception: pass

    print(f"\n  {'✓ All files present' if all_ok else '✗ Some files missing — run: python3 run.py generate'}")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN DISPATCH
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    cmd = sys.argv[1] if len(sys.argv) > 1 else "help"

    if cmd == "generate":
        cmd_generate()

    elif cmd == "grade":
        date_arg = next((a for a in sys.argv[2:] if a.startswith("20")), None)
        no_push  = "--no-push" in sys.argv
        cmd_grade(date_str=date_arg, push=not no_push)

    elif cmd in ("predict", "1", "2", "3", "4", "5"):
        no_push = "--no-push" in sys.argv
        cmd_predict(push=not no_push)

    elif cmd == "fetch-and-predict":
        date_arg = next((a for a in sys.argv[2:] if a.startswith("20")), None)
        no_push  = "--no-push" in sys.argv
        cmd_fetch_and_predict(date_str=date_arg, push=not no_push)

    elif cmd == "train":
        mode = "validate" if "--mode" in sys.argv and sys.argv[sys.argv.index("--mode")+1] == "validate" else "train"
        fast = "--fast" in sys.argv
        cmd_train(mode=mode, fast=fast)

    elif cmd == "generate-dataset":
        cmd_generate_dataset()

    elif cmd == "monitor":
        days = 30
        if "--days" in sys.argv:
            try: days = int(sys.argv[sys.argv.index("--days")+1])
            except Exception: pass
        cmd_monitor(days=days)

    elif cmd in ("token-check", "token", "auth"):
        cmd_token_check()

    elif cmd == "name-test":
        cmd_name_test()

    elif cmd == "check":
        cmd_check()

    else:
        print(__doc__)


if __name__ == "__main__":
    main()
