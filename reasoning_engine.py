"""
PropEdge V2.0 — reasoning_engine.py
────────────────────────────────────────────────────────────────────────────────
Generates pre-match (6-part) and post-match (7-part) plain-English narratives.
Pure functions — takes a dict, returns a string. No I/O, no side effects.

V2.0 additions vs V16:
  - Direction-specific confidence (p_over / p_under separate)
  - H2H home/away edge signal in lead (strongest V2.0 feature)
  - H2H all-agree flag for triple-lock signal
  - line_zscore value classification in narrative
  - Loss type computed fresh from box data (never read from stored field)
────────────────────────────────────────────────────────────────────────────────
"""
from __future__ import annotations
from typing import Optional


def _f(v, default: float = 0.0) -> float:
    try:
        return float(v) if v is not None else default
    except (TypeError, ValueError):
        return default


def _i(v, default: int = 0) -> int:
    try:
        return int(v) if v is not None else default
    except (TypeError, ValueError):
        return default


# ─────────────────────────────────────────────────────────────────────────────
# PRE-MATCH NARRATIVE  (6 parts)
# ─────────────────────────────────────────────────────────────────────────────

def generate_pre_match_reason(play: dict) -> str:
    """
    6-part pre-match narrative for a V2.0 prediction.

    Required play keys:
      player, prop_line, direction, tier, p_over, p_under, confidence_pct,
      predicted_pts, L3, L5, L10, L30, std10, h2h_gap, h2h_avg_pts,
      h2h_games, h2h_valid, h2h_home_away_edge, h2h_all_agree,
      h2h_all_under, is_home, opponent, line_zscore, momentum,
      min_l10, min_l30, h2h_ts_delta, h2h_fga_delta
    """
    name       = play.get("player", "Player")
    line       = _f(play.get("prop_line"), 20)
    direction  = str(play.get("direction") or "OVER").upper()
    direction  = direction if direction in ("OVER", "UNDER") else "OVER"
    tier       = play.get("tier", "PLAY")
    p_over     = _f(play.get("p_over"), 0.5)
    p_under    = _f(play.get("p_under"), 0.5)
    conf       = _f(play.get("confidence_pct"), 60)
    pred_pts   = play.get("predicted_pts")
    is_over    = direction == "OVER"

    L3   = _f(play.get("L3"),   line)
    L5   = _f(play.get("L5"),   L3)
    L10  = _f(play.get("L10"),  L5)
    L30  = _f(play.get("L30"),  L10)
    std10 = _f(play.get("std10"), 5)

    momentum     = L5 - L30
    h2h_gap      = _f(play.get("h2h_gap"), 0)
    h2h_avg      = _f(play.get("h2h_avg_pts"), 0)
    h2h_games    = _i(play.get("h2h_games"), 0)
    h2h_valid    = _i(play.get("h2h_valid"), 0)
    h2h_ha_edge  = _f(play.get("h2h_home_away_edge"), 0)
    h2h_all_agree= _i(play.get("h2h_all_agree"), 0)
    h2h_all_under= _i(play.get("h2h_all_under"), 0)
    is_home      = _i(play.get("is_home"), 0)
    opponent     = play.get("opponent", "") or ""
    line_zscore  = _f(play.get("line_zscore"), 0)
    min_l10      = _f(play.get("min_l10"), 30)
    min_l30      = _f(play.get("min_l30"), 30)
    h2h_ts_delta = _f(play.get("h2h_ts_delta"), 0)

    parts: list[str] = []

    # ── S1: Lead signal — strongest available ─────────────────────────────────
    candidates: list[tuple[float, str]] = []

    # H2H home/away edge (strongest V2.0 signal — corr 0.52)
    if h2h_valid and abs(h2h_ha_edge) >= 1.5:
        venue = "home" if is_home else "away"
        ha_avg = h2h_avg + (h2h_ha_edge - h2h_gap)  # reconstruct venue-specific avg
        direction_word = "above" if h2h_ha_edge > 0 else "below"
        score = abs(h2h_ha_edge) * 1.6
        candidates.append((score,
            f"{name} averages {ha_avg:.1f}pts {venue} vs {opponent} "
            f"({direction_word} the {line} line by {abs(h2h_ha_edge):.1f}pts "
            f"in {h2h_games} H2H matchups)."))

    # Pure H2H gap
    elif h2h_valid and abs(h2h_gap) >= 1.5:
        direction_word = "above" if h2h_gap > 0 else "below"
        candidates.append((abs(h2h_gap) * 1.4,
            f"Against {opponent}, {name} averages {h2h_avg:.1f}pts "
            f"({direction_word} the {line} line by {abs(h2h_gap):.1f}pts "
            f"across {h2h_games} matchups)."))

    # Volume (L30 vs line)
    vol = L30 - line
    if abs(vol) >= 1.5:
        direction_word = "above" if vol > 0 else "below"
        candidates.append((abs(vol) * 1.2,
            f"{name}'s L30 average of {L30:.1f}pts sits {direction_word} "
            f"the {line} line by {abs(vol):.1f}pts."))

    # Momentum
    if abs(momentum) >= 2.5:
        trend = "trending up" if momentum > 0 else "trending down"
        candidates.append((abs(momentum) * 1.0,
            f"{name} is {trend} {abs(momentum):.1f}pts vs L30 baseline "
            f"(L5: {L5:.1f}pts)."))

    # Consistency
    if std10 <= 3.5:
        candidates.append((5.0 - std10,
            f"{name} is highly consistent — std10 of just {std10:.1f}pts "
            f"over last 10 games."))

    if candidates:
        parts.append(sorted(candidates, reverse=True)[0][1])
    else:
        parts.append(f"{name}'s L30 of {L30:.1f}pts vs the {line} line.")

    # ── S2: Line value context (line_zscore) ──────────────────────────────────
    if line_zscore <= -0.5:
        parts.append(
            f"Line value: {line_zscore:.2f}σ below {name}'s L10 average — "
            f"bookmaker has set the line {abs(line_zscore):.1f} standard deviations low (OVER edge).")
    elif line_zscore >= 0.5:
        parts.append(
            f"Line value: {line_zscore:.2f}σ above {name}'s L10 average — "
            f"bookmaker has shaded the line high (UNDER edge).")
    else:
        parts.append(
            f"Line sits close to {name}'s L10 average "
            f"(z-score: {line_zscore:.2f}σ — neutral value).")

    # ── S3: Matchup context ───────────────────────────────────────────────────
    ctx: list[str] = []
    if abs(min_l10 - min_l30) >= 2:
        m_dir = "increasing" if min_l10 > min_l30 else "decreasing"
        ctx.append(f"Minutes {m_dir} (L10: {min_l10:.0f} vs L30: {min_l30:.0f}).")
    if h2h_valid and abs(h2h_ts_delta) > 0.02:
        ts_dir = "better" if h2h_ts_delta > 0 else "worse"
        ctx.append(f"Shooting efficiency runs {ts_dir} vs {opponent} "
                   f"(TS% {h2h_ts_delta:+.1%} vs career).")
    if play.get("opp_b2b"):
        ctx.append(f"{opponent} on back-to-back — fatigued defence.")
    if ctx:
        parts.append(" ".join(ctx[:2]))

    # ── S4: Signal consensus ──────────────────────────────────────────────────
    if h2h_all_agree and is_over:
        parts.append(
            f"Triple-lock: H2H average, L10, and L5 all point OVER the {line} line.")
    elif h2h_all_under and not is_over:
        parts.append(
            f"Triple-lock: H2H average, L10, and L5 all point UNDER the {line} line.")
    elif h2h_valid and (h2h_all_agree or h2h_all_under):
        parts.append(
            f"H2H and rolling windows agree on direction ({direction}).")
    else:
        parts.append(
            f"Model direction: {direction} at {conf:.0f}% confidence [{tier}].")

    # ── S5: Model projection ─────────────────────────────────────────────────
    if pred_pts is not None:
        pred_gap = _f(pred_pts) - line
        sign = "+" if pred_gap >= 0 else ""
        parts.append(
            f"V2.0 model projects {_f(pred_pts):.1f}pts "
            f"({sign}{pred_gap:.1f} vs line). "
            f"P(OVER)={p_over:.0%} | P(UNDER)={p_under:.0%}. "
            f"Confidence: {conf:.0f}% [{tier}].")

    # ── S6: Risk note ─────────────────────────────────────────────────────────
    risk = None
    if std10 > 7:
        risk = f"[High variance: std10={std10:.1f}pts — outcome can deviate significantly.]"
    elif is_over and momentum > 6:
        risk = (f"[Reversion risk: L5 is {momentum:+.1f}pts above L30 — "
                f"partial mean reversion is possible.]")
    elif not is_over and momentum < -6:
        risk = (f"[Bounce-back risk: player is in a {abs(momentum):.1f}pt cold spell — "
                f"regression to mean possible.]")
    elif _i(play.get("opp_b2b"), 0) == 0 and play.get("post_allstar_flag") == 0 and std10 > 5:
        risk = "[Moderate variance — monitor minutes for last-minute lineup changes.]"
    if risk:
        parts.append(risk)

    return " ".join(p for p in parts if p)


# ─────────────────────────────────────────────────────────────────────────────
# POST-MATCH NARRATIVE  (7 parts)
# ─────────────────────────────────────────────────────────────────────────────

def generate_post_match_reason(
    play: dict,
    box_data: Optional[dict] = None,
) -> tuple[str, str]:
    """
    7-part post-match narrative.
    Returns (narrative_string, loss_type_string).

    loss_type is always computed fresh from box_data — never read from stored
    field (avoids stale classification bug from V9.x).
    """
    name      = play.get("player", "Player")
    line      = _f(play.get("prop_line"), 20)
    direction = str(play.get("direction") or "OVER").upper()
    direction = direction if direction in ("OVER", "UNDER") else "OVER"
    pred_pts  = play.get("predicted_pts")
    p_over    = _f(play.get("p_over"), 0.5)
    tier      = play.get("tier", "PLAY")
    is_over   = direction == "OVER"
    opponent  = play.get("opponent", "this opponent") or "this opponent"

    box = box_data or {}
    actual_pts = _f(box.get("actual_pts",  play.get("actual_pts")),  0)
    actual_min = _f(box.get("actual_min",  play.get("actual_min")),  0)
    actual_fga = _f(box.get("actual_fga"),  0)
    actual_fgm = _f(box.get("actual_fgm"),  0)
    actual_fg  = actual_fgm / max(actual_fga, 1)

    min_l10 = _f(play.get("min_l10"), 30)
    fga_l10 = _f(play.get("fga_l10"), 10)
    fg_l10  = _f(play.get("fg_pct_l10"), 0.45)
    momentum= _f(play.get("momentum"), 0)
    h2h_gap = _f(play.get("h2h_gap"), 0)
    h2h_valid=_i(play.get("h2h_valid"), 0)
    conf    = _f(play.get("confidence_pct"), 60)

    delta  = actual_pts - line
    margin = abs(delta)
    won    = (is_over and actual_pts > line) or (not is_over and actual_pts < line)
    integrity_flag = box.get("integrity_flag", "")

    # ── Loss type (computed fresh — not from stored field) ────────────────────
    loss_type = "MODEL_CORRECT"
    if not won:
        if actual_min > 0 and actual_min < min_l10 - 4:
            loss_type = "MINUTES_SHORTFALL"
        elif actual_fga > 0 and abs(actual_fg - fg_l10) >= 0.08:
            loss_type = "SHOOTING_VARIANCE"
        elif margin <= 1.5:
            loss_type = "CLOSE_CALL"
        elif margin > 9:
            loss_type = "BLOWOUT_EFFECT"
        elif abs(momentum) > 5:
            loss_type = "TREND_REVERSAL"
        elif h2h_valid and abs(h2h_gap) >= 4 and (
            (is_over and h2h_gap > 0) or (not is_over and h2h_gap < 0)
        ):
            loss_type = "H2H_OVERRIDE_FAILURE"
        elif conf >= 88:
            loss_type = "HIGH_CONF_ANOMALY"
        else:
            loss_type = "MODEL_FAILURE_GENERAL"

    parts: list[str] = []

    # S1: Outcome
    outcome = "HIT ✓" if won else "MISSED ✗"
    over_under = "over" if delta > 0 else ("exactly" if delta == 0 else "under")
    parts.append(
        f"{outcome} — {name} scored {actual_pts:.0f}pts vs {line} line "
        f"({over_under} by {margin:.1f}pts, called {direction})."
    )

    # S2: Minutes vs expectation
    if actual_min > 0:
        min_diff = actual_min - min_l10
        if abs(min_diff) >= 3:
            more_less = "more" if min_diff > 0 else "fewer"
            parts.append(
                f"Minutes: {actual_min:.0f} played vs {min_l10:.0f} L10 average "
                f"({more_less} than expected by {abs(min_diff):.0f}min)."
            )

    # S3: Shooting efficiency
    if actual_fga > 0 and abs(actual_fg - fg_l10) >= 0.06:
        temp = "hot" if actual_fg > fg_l10 else "cold"
        parts.append(
            f"Shot efficiency: {actual_fg:.0%} FG% vs {fg_l10:.0%} L10 average — "
            f"{temp} shooting."
        )

    # S4: Model accuracy
    if pred_pts is not None:
        model_err  = abs(actual_pts - _f(pred_pts))
        model_correct = won
        parts.append(
            f"Model projection: {_f(pred_pts):.1f}pts "
            f"(error: {model_err:.1f}pts, direction {'correct' if model_correct else 'incorrect'}). "
            f"P(OVER)={p_over:.0%} | Confidence: {conf:.0f}% [{tier}]."
        )

    # S5: H2H context (V2.0 addition)
    if h2h_valid:
        direction_word = "aligned" if (
            (is_over and h2h_gap > 0) or (not is_over and h2h_gap < 0)
        ) else "opposed"
        parts.append(
            f"H2H signal was {direction_word} with model call "
            f"(H2H avg gap vs line: {h2h_gap:+.1f}pts)."
        )

    # S6: Loss classification
    _loss_desc = {
        "MODEL_CORRECT":          "Prediction confirmed. Model and H2H validated.",
        "CLOSE_CALL":             f"Margin of {margin:.1f}pts — near-boundary result. Statistical noise at the line.",
        "MINUTES_SHORTFALL":      f"Player received {actual_min:.0f} min vs expected {min_l10:.0f} — usage restricted.",
        "SHOOTING_VARIANCE":      f"FG% deviated {abs(actual_fg - fg_l10):.0%} from L10 norm — efficiency shock.",
        "BLOWOUT_EFFECT":         f"Margin of {margin:.0f}pts suggests game script disrupted normal patterns.",
        "TREND_REVERSAL":         f"Momentum of {momentum:+.1f}pts vs baseline — mean reversion occurred.",
        "H2H_OVERRIDE_FAILURE":   f"Strong H2H signal ({h2h_gap:+.1f}pt gap) did not materialise — opponent-specific anomaly.",
        "HIGH_CONF_ANOMALY":      f"High-confidence call ({conf:.0f}%) failed — possible lineup or situational shock.",
        "MODEL_FAILURE_GENERAL":  "No dominant structural cause — genuine uncertainty at this confidence level.",
    }
    parts.append(_loss_desc.get(loss_type, ""))

    # S7: Learning note
    if loss_type == "CLOSE_CALL":
        parts.append(f"Sub-2pt margin vs {opponent} — consider as lean, not lock, in future.")
    elif loss_type == "MINUTES_SHORTFALL":
        parts.append(f"Monitor {name}'s rotation — minutes restriction may be ongoing.")
    elif loss_type == "TREND_REVERSAL":
        parts.append(f"Extreme momentum ({momentum:+.1f}pt) plays carry reversion risk — confidence penalised automatically.")
    elif loss_type == "H2H_OVERRIDE_FAILURE":
        parts.append(f"H2H edge vs {opponent} weakened — reassess historical matchup pattern.")
    elif won:
        parts.append(f"Model and H2H alignment validated vs {opponent}.")

    if integrity_flag:
        parts.append(f"⚠ Data note: {integrity_flag}")

    return " ".join(p for p in parts if p), loss_type
