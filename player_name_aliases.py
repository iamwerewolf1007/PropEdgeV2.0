"""
PropEdge V2.0 — player_name_aliases.py
────────────────────────────────────────────────────────────────────────────────
Single source of truth for player name resolution across the entire pipeline.
Used by: batch_predict.py, batch_grade.py, generate_dataset.py

Resolves Odds-API / Excel player names → canonical NBA game-log PLAYER_NAME.

HOW TO UPDATE:
  1. Add entry: "normalised_odds_api_name": "Exact NBA CSV Name"
  2. Run: python3 player_name_aliases.py   ← self-test, all cases must pass
────────────────────────────────────────────────────────────────────────────────
"""
from __future__ import annotations
import re
import unicodedata

_SUFFIX_RE = re.compile(r"\s+(jr\.?|sr\.?|ii|iii|iv)$", re.IGNORECASE)


def _norm(s: str) -> str:
    """Strip accents, punctuation, lowercase — canonical comparison key."""
    n = unicodedata.normalize("NFD", str(s))
    n = "".join(c for c in n if unicodedata.category(c) != "Mn")
    return re.sub(r"[^a-z0-9 ]", "", n.lower()).strip()


def _norm_strip(s: str) -> str:
    """_norm + remove Jr/Sr/II/III/IV suffix."""
    return _SUFFIX_RE.sub("", _norm(s)).strip()


PLAYER_ALIASES: dict[str, str] = {

    # ── Nicknames / common-name mismatches ────────────────────────────────────
    "carlton carrington":       "Bub Carrington",
    "moe wagner":               "Moritz Wagner",
    "cam reddish":              "Cameron Reddish",
    "james huff":               "Jay Huff",
    "nicolas claxton":          "Nic Claxton",
    "mohamed bamba":            "Mo Bamba",
    "herb jones":               "Herbert Jones",
    "naz reid":                 "Naz Reid",
    "og anunoby":               "O.G. Anunoby",

    # ── Suffix mismatches ─────────────────────────────────────────────────────
    "ron holland":              "Ronald Holland II",
    "vincent williams jr":      "Vince Williams Jr.",
    "paul reed jr":             "Paul Reed",
    "bruce brown jr":           "Bruce Brown",
    "isaiah stewart ii":        "Isaiah Stewart",
    "derrick jones":            "Derrick Jones Jr.",
    "bj boston jr":             "Brandon Boston Jr.",
    "bj boston":                "Brandon Boston Jr.",

    # ── Accented / international names ───────────────────────────────────────
    "nikola jokic":             "Nikola Jokić",
    "luka doncic":              "Luka Dončić",
    "kristaps porzingis":       "Kristaps Porziņģis",
    "bojan bogdanovic":         "Bojan Bogdanović",
    "bogdan bogdanovic":        "Bogdan Bogdanović",
    "dario saric":              "Dario Šarić",
    "alperen sengun":           "Alperen Şengün",
    "nikola vucevic":           "Nikola Vučević",
    "dennis schroder":          "Dennis Schröder",
    "lester quinones":          "Lester Quiñones",
    "vit krejci":               "Vít Krejčí",
    "vasilije micic":           "Vasilije Mičić",
    "tristan vukcevic":         "Tristan Vukčević",
    "jonas valanciunas":        "Jonas Valančiūnas",
    "moussa diabate":           "Moussa Diabaté",
    "tidjane salaun":           "Tidjane Salaün",
    "dante exum":               "Danté Exum",
    "hugo gonzalez":            "Hugo González",
    "jusuf nurkic":             "Jusuf Nurkić",
    "karlo matkovic":           "Karlo Matković",
    "kasparas jakucionis":      "Kasparas Jakučionis",
    "monte morris":             "Monté Morris",
    "nikola jovic":             "Nikola Jović",
    "yanic konan niederhauser": "Yanic Konan Niederhäuser",
    "egor demin":               "Egor Dëmin",
    "vlatko cancar":            "Vlatko Čančar",

    # ── Jr. — trailing period missing in Odds API ─────────────────────────────
    "andre jackson jr":         "Andre Jackson Jr.",
    "craig porter jr":          "Craig Porter Jr.",
    "gary trent jr":            "Gary Trent Jr.",
    "jabari smith jr":          "Jabari Smith Jr.",
    "jaime jaquez jr":          "Jaime Jaquez Jr.",
    "jaren jackson jr":         "Jaren Jackson Jr.",
    "jeff dowtin jr":           "Jeff Dowtin Jr.",
    "kelly oubre jr":           "Kelly Oubre Jr.",
    "kenyon martin jr":         "Kenyon Martin Jr.",
    "keion brooks jr":          "Keion Brooks Jr.",
    "kevin porter jr":          "Kevin Porter Jr.",
    "larry nance jr":           "Larry Nance Jr.",
    "michael porter jr":        "Michael Porter Jr.",
    "nick smith jr":            "Nick Smith Jr.",
    "scotty pippen jr":         "Scotty Pippen Jr.",
    "terrence shannon jr":      "Terrence Shannon Jr.",
    "tim hardaway jr":          "Tim Hardaway Jr.",
    "wendell carter jr":        "Wendell Carter Jr.",
    "wendell moore jr":         "Wendell Moore Jr.",

    # ── II / III / IV / Sr ────────────────────────────────────────────────────
    "dereck lively ii":         "Dereck Lively II",
    "gary payton ii":           "Gary Payton II",
    "lindy waters iii":         "Lindy Waters III",
    "marvin bagley iii":        "Marvin Bagley III",
    "ricky council iv":         "Ricky Council IV",
    "trey murphy iii":          "Trey Murphy III",
    "xavier tillman sr":        "Xavier Tillman Sr.",

    # ── Hyphenated — Odds API strips hyphen ───────────────────────────────────
    "shai gilgeousalexander":   "Shai Gilgeous-Alexander",
    "dorian finneysmith":       "Dorian Finney-Smith",
    "talen hortontucker":       "Talen Horton-Tucker",
    "kentavious caldwellpope":  "Kentavious Caldwell-Pope",
    "nickeil alexanderwalker":  "Nickeil Alexander-Walker",
    "jalen hoodschifino":       "Jalen Hood-Schifino",
    "jeremiah robinsonearl":    "Jeremiah Robinson-Earl",
    "trayce jacksondavis":      "Trayce Jackson-Davis",
    "oliviermaxence prosper":   "Olivier-Maxence Prosper",

    # ── Dot abbreviations ─────────────────────────────────────────────────────
    "aj green":                 "A.J. Green",
    "aj lawson":                "A.J. Lawson",
    "cj mccollum":              "C.J. McCollum",
    "gg jackson":               "G.G. Jackson",
    "kj martin":                "K.J. Martin",
    "pj washington":            "P.J. Washington",
    "rj barrett":               "R.J. Barrett",
    "tj mcconnell":             "T.J. McConnell",

    # ── Apostrophe names ──────────────────────────────────────────────────────
    "dayron sharpe":            "Day'Ron Sharpe",
    "deaaron fox":              "De'Aaron Fox",
    "deandre hunter":           "De'Andre Hunter",
    "deanthony melton":         "De'Anthony Melton",
    "jaesean tate":             "Jae'Sean Tate",
    "jakobe walter":            "Ja'Kobe Walter",
    "kelel ware":               "Kel'el Ware",
    "royce oneale":             "Royce O'Neale",
    "dangelo russell":          "D'Angelo Russell",
}


def resolve_name(player_raw: str, nmap: dict[str, str]) -> str | None:
    """
    Resolve an Odds-API player name → canonical NBA game-log name.

    nmap = {_norm(csv_name): csv_name}  built once per run.

    Resolution order:
      1. PLAYER_ALIASES  (explicit, human-verified — highest priority)
      2. Exact normalised match
      3. Suffix-stripped exact
      4. 8-char prefix match
      5. Token overlap >= 80%
    """
    key          = _norm(player_raw)
    key_stripped = _norm_strip(player_raw)

    # 1. Alias table
    alias_target = PLAYER_ALIASES.get(key) or PLAYER_ALIASES.get(key_stripped)
    if alias_target:
        av = _norm(alias_target)
        if av in nmap:
            return nmap[av]
        for k, v in nmap.items():
            if _norm(v) == av:
                return v

    # 2. Exact normalised
    if key in nmap:
        return nmap[key]

    # 3. Suffix-stripped exact
    if key_stripped in nmap:
        return nmap[key_stripped]

    # 4. 8-char prefix
    if len(key) >= 6:
        for k, v in nmap.items():
            if len(k) >= 6 and key[:8] == k[:8]:
                return v

    # 5. Suffix-stripped prefix-8
    if key_stripped != key and len(key_stripped) >= 6:
        for k, v in nmap.items():
            if len(k) >= 6 and key_stripped[:8] == k[:8]:
                return v

    # 6. Token overlap >= 80%
    key_tokens = set(key_stripped.split())
    best_score, best_match = 0.0, None
    for k, v in nmap.items():
        ct = set(_norm_strip(v).split())
        if not ct:
            continue
        score = len(key_tokens & ct) / max(len(key_tokens), len(ct))
        if score > best_score:
            best_score, best_match = score, v
    if best_score >= 0.80:
        return best_match

    return None


def resolve_grade_name(player_raw: str, box: dict[str, float]) -> float | None:
    """
    Resolve player name against box score dict {normalised_name: pts}.
    Returns actual_pts or None if unresolvable.
    """
    key          = _norm(player_raw)
    key_stripped = _norm_strip(player_raw)

    alias_target = PLAYER_ALIASES.get(key) or PLAYER_ALIASES.get(key_stripped)
    if alias_target:
        av = _norm(alias_target)
        if av in box:
            return box[av]

    if key in box:
        return box[key]
    if key_stripped in box:
        return box[key_stripped]

    if len(key) >= 6:
        for bkey, pts in box.items():
            if len(bkey) >= 6 and key[:8] == bkey[:8]:
                return pts

    key_tokens = set(key_stripped.split())
    best_score, best_pts = 0.0, None
    for bkey, pts in box.items():
        ct = set(bkey.split())
        if not ct:
            continue
        score = len(key_tokens & ct) / max(len(key_tokens), len(ct))
        if score > best_score:
            best_score, best_pts = score, pts
    if best_score >= 0.80:
        return best_pts

    return None


def build_nmap(player_names: list[str]) -> dict[str, str]:
    """Build normalised lookup map from a list of canonical CSV names."""
    return {_norm(n): n for n in player_names}


# ─────────────────────────────────────────────────────────────────────────────
# SELF-TEST  (python3 player_name_aliases.py)
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    fake_csv = [
        "Bub Carrington", "Moritz Wagner", "Cameron Reddish", "Jay Huff",
        "Nic Claxton", "Mo Bamba", "Herbert Jones", "Ronald Holland II",
        "Vince Williams Jr.", "Paul Reed", "Bruce Brown", "Isaiah Stewart",
        "Derrick Jones Jr.", "Brandon Boston Jr.",
        "Nikola Jokić", "Luka Dončić", "Kristaps Porziņģis", "Alperen Şengün",
        "Nikola Vučević", "Dennis Schröder", "Jonas Valančiūnas",
        "Moussa Diabaté", "Tidjane Salaün", "Danté Exum",
        "Andre Jackson Jr.", "Gary Trent Jr.", "Jaren Jackson Jr.",
        "Kelly Oubre Jr.", "Tim Hardaway Jr.", "Wendell Carter Jr.",
        "Dereck Lively II", "Gary Payton II", "Trey Murphy III", "Ricky Council IV",
        "Shai Gilgeous-Alexander", "Dorian Finney-Smith",
        "Kentavious Caldwell-Pope", "Trayce Jackson-Davis",
        "A.J. Green", "A.J. Lawson", "C.J. McCollum", "P.J. Washington",
        "R.J. Barrett", "T.J. McConnell", "G.G. Jackson", "K.J. Martin",
        "Day'Ron Sharpe", "De'Aaron Fox", "Royce O'Neale", "D'Angelo Russell",
        "Jae'Sean Tate", "Kel'el Ware", "O.G. Anunoby",
        "LeBron James", "Stephen Curry", "Kevin Durant",
    ]
    nmap = build_nmap(fake_csv)

    cases = [
        ("Carlton Carrington",      "Bub Carrington"),
        ("Moe Wagner",              "Moritz Wagner"),
        ("Cam Reddish",             "Cameron Reddish"),
        ("James Huff",              "Jay Huff"),
        ("Nicolas Claxton",         "Nic Claxton"),
        ("Mohamed Bamba",           "Mo Bamba"),
        ("Herb Jones",              "Herbert Jones"),
        ("Ron Holland",             "Ronald Holland II"),
        ("Vincent Williams Jr",     "Vince Williams Jr."),
        ("Paul Reed Jr",            "Paul Reed"),
        ("Bruce Brown Jr",          "Bruce Brown"),
        ("Isaiah Stewart II",       "Isaiah Stewart"),
        ("Derrick Jones",           "Derrick Jones Jr."),
        ("BJ Boston Jr",            "Brandon Boston Jr."),
        ("Nikola Jokic",            "Nikola Jokić"),
        ("Luka Doncic",             "Luka Dončić"),
        ("Kristaps Porzingis",      "Kristaps Porziņģis"),
        ("Alperen Sengun",          "Alperen Şengün"),
        ("Andre Jackson Jr",        "Andre Jackson Jr."),
        ("Gary Trent Jr",           "Gary Trent Jr."),
        ("Jaren Jackson Jr",        "Jaren Jackson Jr."),
        ("Kelly Oubre Jr",          "Kelly Oubre Jr."),
        ("Tim Hardaway Jr",         "Tim Hardaway Jr."),
        ("Wendell Carter Jr",       "Wendell Carter Jr."),
        ("Dereck Lively II",        "Dereck Lively II"),
        ("Gary Payton II",          "Gary Payton II"),
        ("Trey Murphy III",         "Trey Murphy III"),
        ("Ricky Council IV",        "Ricky Council IV"),
        ("Shai Gilgeous-Alexander", "Shai Gilgeous-Alexander"),
        ("Dorian Finney-Smith",     "Dorian Finney-Smith"),
        ("A.J. Green",              "A.J. Green"),
        ("AJ Green",                "A.J. Green"),
        ("AJ Lawson",               "A.J. Lawson"),
        ("R.J. Barrett",            "R.J. Barrett"),
        ("Day'Ron Sharpe",          "Day'Ron Sharpe"),
        ("De'Aaron Fox",            "De'Aaron Fox"),
        ("Royce O'Neale",           "Royce O'Neale"),
        ("Jae'Sean Tate",           "Jae'Sean Tate"),
        ("OG Anunoby",              "O.G. Anunoby"),
        ("LeBron James",            "LeBron James"),
    ]

    print(f"\n  player_name_aliases.py — self-test ({len(cases)} cases)")
    print(f"  {'Input':<35} {'Expected':<32} {'Got':<32} OK")
    print("  " + "─" * 105)
    passed = failed = 0
    for excel, expected in cases:
        got = resolve_name(excel, nmap)
        ok  = (got == expected)
        if ok:
            passed += 1
        else:
            failed += 1
            print(f"  {excel:<35} {expected:<32} {str(got):<32}  ❌")
    print()
    if failed == 0:
        print(f"  ✅  {passed}/{len(cases)} passed")
    else:
        print(f"  ❌  {passed}/{len(cases)} passed  ({failed} failed — fix before deploying)")
    print(f"  Alias entries: {len(PLAYER_ALIASES)}")
