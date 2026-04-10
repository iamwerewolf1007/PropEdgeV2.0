"""
PropEdge V2.0 — fetch_props.py
Fetch today's NBA player points props from the Odds API.
Returns games dict consumed directly by batch_predict.run_predict().

RUN: python3 run.py fetch-and-predict
"""
from __future__ import annotations

import time
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

from config import ODDS_API_KEY, ODDS_API_BASE, ODDS_SPORT, ODDS_MARKET, CREDIT_ALERT, VERSION_TAG

ET = ZoneInfo("America/New_York")

_TEAM_ABR: dict[str, str] = {
    "Atlanta Hawks": "ATL", "Boston Celtics": "BOS", "Brooklyn Nets": "BKN",
    "Charlotte Hornets": "CHA", "Chicago Bulls": "CHI", "Cleveland Cavaliers": "CLE",
    "Dallas Mavericks": "DAL", "Denver Nuggets": "DEN", "Detroit Pistons": "DET",
    "Golden State Warriors": "GSW", "Houston Rockets": "HOU", "Indiana Pacers": "IND",
    "LA Clippers": "LAC", "Los Angeles Clippers": "LAC", "Los Angeles Lakers": "LAL",
    "LA Lakers": "LAL", "Memphis Grizzlies": "MEM", "Miami Heat": "MIA",
    "Milwaukee Bucks": "MIL", "Minnesota Timberwolves": "MIN", "New Orleans Pelicans": "NOP",
    "New York Knicks": "NYK", "Oklahoma City Thunder": "OKC", "Orlando Magic": "ORL",
    "Philadelphia 76ers": "PHI", "Phoenix Suns": "PHX", "Portland Trail Blazers": "POR",
    "Sacramento Kings": "SAC", "San Antonio Spurs": "SAS", "Toronto Raptors": "TOR",
    "Utah Jazz": "UTA", "Washington Wizards": "WAS",
}

def _abr(t): return _TEAM_ABR.get(t, t[:3].upper())


def _api_get(url: str, params: dict, timeout: int = 30):
    try:
        import requests as _r
        resp = _r.get(url, params=params, timeout=timeout)
        resp.raise_for_status()
        return resp.json(), dict(resp.headers)
    except ImportError:
        import urllib.request, urllib.parse, ssl, json as _j
        qs  = urllib.parse.urlencode(params)
        ctx = ssl.create_default_context()
        try:
            import certifi
            ctx = ssl.create_default_context(cafile=certifi.where())
        except ImportError:
            pass
        with urllib.request.urlopen(f"{url}?{qs}", timeout=timeout, context=ctx) as r:
            return _j.loads(r.read()), dict(r.headers)


def _credits(headers: dict, label: str = "") -> None:
    n = headers.get("x-requests-remaining", headers.get("X-Requests-Remaining", "?"))
    try:
        ni = int(n)
        print(f"    Credits: {ni}{'  ⚠ LOW' if ni <= CREDIT_ALERT else ''}  {label}")
    except (ValueError, TypeError):
        pass


def fetch_props_for_predict(date_str: str | None = None) -> dict[str, dict]:
    """
    Fetch NBA player props for the given ET date.
    Returns {event_id: {home, away, home_raw, away_raw, gametime, commence, props: {player: {...}}}}
    Consumed by batch_predict.run_predict(games=...).
    """
    if date_str is None:
        date_str = datetime.now(ET).strftime("%Y-%m-%d")

    d   = datetime.strptime(date_str, "%Y-%m-%d")
    frm = (d - timedelta(hours=6)).strftime("%Y-%m-%dT%H:%M:%SZ")
    to  = (d + timedelta(hours=30)).strftime("%Y-%m-%dT%H:%M:%SZ")

    print(f"\n  {VERSION_TAG} — Fetch Props  [{date_str}]")
    print("  " + "─" * 52)
    print(f"  Fetching events...")

    body, hdrs = _api_get(
        f"{ODDS_API_BASE}/sports/{ODDS_SPORT}/events",
        {"apiKey": ODDS_API_KEY, "dateFormat": "iso",
         "commenceTimeFrom": frm, "commenceTimeTo": to},
    )
    _credits(hdrs, "events")

    events = []
    for e in body:
        ts = e.get("commence_time", "")
        try:
            utc_dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
            et_dt  = utc_dt.astimezone(ET)
            if et_dt.strftime("%Y-%m-%d") == date_str:
                e["_gametime"] = et_dt.strftime("%-I:%M %p ET")
                events.append(e)
        except Exception:
            continue

    print(f"  {len(events)} games on {date_str} ET")
    if not events:
        return {}

    games: dict[str, dict] = {}
    for e in events:
        eid = e["id"]
        games[eid] = {
            "home":     _abr(e.get("home_team", "")),
            "away":     _abr(e.get("away_team", "")),
            "home_raw": e.get("home_team", ""),
            "away_raw": e.get("away_team", ""),
            "gametime": e.get("_gametime", ""),
            "commence": e.get("commence_time", ""),
            "props":    {},
            "_raw_lines": {},
        }

    print(f"\n  Fetching odds for {len(games)} games...")
    for eid, g in games.items():
        time.sleep(0.3)
        try:
            body2, hdrs2 = _api_get(
                f"{ODDS_API_BASE}/sports/{ODDS_SPORT}/events/{eid}/odds",
                {"apiKey": ODDS_API_KEY, "regions": "us",
                 "markets": f"{ODDS_MARKET},spreads,totals",
                 "oddsFormat": "american", "dateFormat": "iso"},
            )
            _credits(hdrs2)
            for bm in body2.get("bookmakers", []):
                for mkt in bm.get("markets", []):
                    if mkt.get("key") != ODDS_MARKET:
                        continue
                    for o in mkt.get("outcomes", []):
                        player = (o.get("description") or "").strip() or o.get("name", "").strip()
                        pt     = o.get("point")
                        side   = (o.get("name") or "").upper()
                        price  = o.get("price")
                        if not player or pt is None:
                            continue
                        g["_raw_lines"].setdefault(player, []).append(float(pt))
                        if player not in g["props"]:
                            g["props"][player] = {"line": float(pt), "over": None, "under": None,
                                                   "books": 0, "min_line": float(pt), "max_line": float(pt)}
                        pd_ = g["props"][player]
                        pd_["min_line"] = min(pd_["min_line"], float(pt))
                        pd_["max_line"] = max(pd_["max_line"], float(pt))
                        if side == "OVER":
                            pd_["over"] = int(price) if price else -110
                            pd_["books"] += 1
                        elif side == "UNDER":
                            pd_["under"] = int(price) if price else -110
            n_props = len(g["props"])
            print(f"    ✓ {g['away']} @ {g['home']}: {n_props} props")
        except Exception as ex:
            print(f"    ✗ {g.get('away_raw','')} @ {g.get('home_raw','')}: {ex}")
            time.sleep(1)

    total = sum(len(g["props"]) for g in games.values())
    active = sum(1 for g in games.values() if g["props"])
    print(f"\n  ✓ {total} props across {active}/{len(games)} games")
    return games
