"""The decision engine: expected WP of going for it, kicking, or punting.

For a 4th down the engine first builds three *WP curves* for the offense,
using the WP model on every state the play could lead to:

* ``own[y]``   offense has 1st-and-10 (or goal) at yardline ``y``      (y = 1..99)
* ``opp[y]``   opponent has 1st-and-10 at *its* yardline ``y``         (y = 1..99)
* ``score[k]`` offense just scored 3 / 6 / 7 points and kicks off

All curves are on the 4th-down offense's WP scale (opponent states are read
as ``1 - wp``). Each action's expected WP is then plain arithmetic over those
curves and the sub-model outcome distributions (:func:`action_values`), so
re-running the arithmetic under bootstrapped sub-models is almost free.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .data import STATE_COLS
from .state import (advance_clock, first_down, flip_possession, kickoff_to_opponent,
                    resolve_halftime, terminal_wp)
from .submodels import SubModels

ACTIONS = ("go", "fg", "punt")
YARDS = np.arange(1, 100)
SCORES = (3, 6, 7)
PLAY_SECONDS = 6.0  # clock used by any 4th-down snap (a simplification; see README)


@dataclass
class Curves:
    own: np.ndarray    # (n, 99)
    opp: np.ndarray    # (n, 99)
    score: np.ndarray  # (n, 3) for SCORES


def state_wp(states: pd.DataFrame, flipped: np.ndarray, wp_fn, kickoff_yl) -> np.ndarray:
    """WP of ``states`` for the 4th-down offense.

    ``flipped`` marks states whose offense is the *opponent*. Handles halftime
    (2nd-half receiver gets the ball) and the end of regulation (final score).
    """
    s, swapped = resolve_halftime(states.reset_index(drop=True), kickoff_yl)
    over = (s["game_seconds_remaining"] <= 0).to_numpy()
    wp = np.empty(len(s))
    if over.any():
        wp[over] = terminal_wp(s.loc[over, "score_diff"])
    if (~over).any():
        wp[~over] = wp_fn(s.loc[~over])
    return np.where(np.asarray(flipped) ^ swapped, 1 - wp, wp)


def build_curves(plays: pd.DataFrame, wp_fn, kickoff_yl, seconds: float = PLAY_SECONDS) -> Curves:
    """Evaluate the WP model on every post-snap state for each play."""
    n = len(plays)
    base = advance_clock(plays[STATE_COLS].reset_index(drop=True).astype(float), seconds)
    ko = np.broadcast_to(np.asarray(kickoff_yl, dtype=float), (n,))
    rep = base.loc[base.index.repeat(len(YARDS))].reset_index(drop=True)
    y = np.tile(YARDS, n).astype(float)
    ko_rep = np.repeat(ko, len(YARDS))
    own = state_wp(first_down(rep, y), np.zeros(len(rep), bool), wp_fn, ko_rep).reshape(n, -1)
    opp = state_wp(flip_possession(rep, y), np.ones(len(rep), bool), wp_fn, ko_rep).reshape(n, -1)
    score = np.column_stack([
        state_wp(kickoff_to_opponent(base, k, ko), np.ones(n, bool), wp_fn, ko) for k in SCORES
    ])
    return Curves(own, opp, score)


def _lookup(curve: np.ndarray, yl: np.ndarray) -> np.ndarray:
    """Row-wise curve value at (rounded, clipped) yardlines; ``yl`` is (n,) or (n, k)."""
    idx = np.clip(np.rint(yl).astype(int), 1, 99) - 1
    rows = np.arange(curve.shape[0])
    return curve[rows[:, None], idx] if idx.ndim == 2 else curve[rows, idx]


def action_values(yardline_100, ydstogo, curves: Curves, p_convert, extra_yards,
                  p_fg, punt_opp_yl, p_xp) -> np.ndarray:
    """Expected WP of (go, fg, punt) for each play; returns an (n, 3) array.

    * go:   p * E[WP | converted] + (1 - p) * WP(opponent ball at the spot)
    * fg:   p * WP(+3, kickoff) + (1 - p) * WP(opponent ball at spot of kick, or its 20)
    * punt: E[WP(opponent ball at the punt result)]

    ``extra_yards`` (n, k) and ``punt_opp_yl`` (n, m) are equally weighted
    outcome draws. A conversion reaching the end zone is a touchdown, worth
    7 with probability ``p_xp`` and 6 otherwise.
    """
    yl = np.asarray(yardline_100, float)
    ytg = np.asarray(ydstogo, float)
    w3, w6, w7 = curves.score.T
    wp_td = p_xp * w7 + (1 - p_xp) * w6

    new_yl = yl[:, None] - ytg[:, None] - np.asarray(extra_yards, float)
    td = (new_yl <= 0) | (ytg >= yl)[:, None]
    succ = np.where(td, wp_td[:, None], _lookup(curves.own, np.maximum(new_yl, 1))).mean(1)
    fail = _lookup(curves.opp, 100 - yl)
    go = p_convert * succ + (1 - p_convert) * fail

    miss = _lookup(curves.opp, np.minimum(80, 93 - yl))
    fg = p_fg * w3 + (1 - p_fg) * miss

    punt = _lookup(curves.opp, np.asarray(punt_opp_yl, float)).mean(1)
    return np.column_stack([go, fg, punt])


def values_from_models(plays: pd.DataFrame, curves: Curves, sm: SubModels, p_xp) -> np.ndarray:
    """:func:`action_values` with outcome distributions taken from fitted sub-models."""
    return action_values(plays["yardline_100"].to_numpy(), plays["ydstogo"].to_numpy(), curves,
                         sm.conversion.predict(plays), sm.gain.predict(plays),
                         sm.field_goal.predict(plays), sm.punt.predict(plays), np.asarray(p_xp, float))


def recommend(values: np.ndarray) -> pd.DataFrame:
    """Best action, its WP, and the gap to the runner-up for an (n, 3) value array."""
    order = np.argsort(-values, axis=1)
    best = order[:, 0]
    rows = np.arange(len(values))
    return pd.DataFrame({
        "best": np.array(ACTIONS)[best],
        "wp_best": values[rows, best],
        "gap": values[rows, best] - values[rows, order[:, 1]],
    })


def bootstrap_confidence(plays: pd.DataFrame, curves: Curves, boot_models: list[SubModels],
                         p_xp, point_best: np.ndarray) -> pd.DataFrame:
    """Share of bootstrap sub-model refits that agree with the point recommendation.

    The WP curves are held fixed, so this measures uncertainty from the outcome
    sub-models (conversion, field goal, punt, gain), not from the WP model.
    """
    col = np.array([ACTIONS.index(a) for a in point_best])
    rows = np.arange(len(plays))
    margins = []
    for sm in boot_models:
        v = values_from_models(plays, curves, sm, p_xp)
        other = v.copy()
        other[rows, col] = -np.inf
        margins.append(v[rows, col] - other.max(1))  # < 0: this refit prefers another call
    m = np.array(margins)
    return pd.DataFrame({"confidence": (m > 0).mean(0), "gap_lo": np.quantile(m, 0.05, axis=0),
                         "gap_hi": np.quantile(m, 0.95, axis=0)})
