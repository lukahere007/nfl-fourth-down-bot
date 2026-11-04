"""Game-state algebra: possession flips, clock advance and WP features.

A *state* is a DataFrame with the columns in :data:`STATE_COLS`, always written
from the point of view of the team with the ball (the offense):

* ``score_diff``  offense score minus defense score
* ``yardline_100`` yards from the opponent's end zone (1 = opponent 1-yard line)
* ``spread``      pre-game expected margin for the offense (Vegas prior)
* ``receive_2h``  1 if the offense receives the second-half kickoff (1st half only)
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .data import STATE_COLS

HALF = 1800.0
GAME = 3600.0


def flip_possession(state: pd.DataFrame, yardline_100: np.ndarray | float | None = None) -> pd.DataFrame:
    """Return the state from the *other* team's point of view, as 1st-and-10.

    ``yardline_100`` is the new offense's distance to goal. If omitted, the ball
    stays at the same spot (``100 - yardline_100``), e.g. a turnover on downs.
    Goal-to-go is handled by capping ``ydstogo`` at the distance to goal.
    """
    s = state.copy()
    new_yl = 100 - state["yardline_100"] if yardline_100 is None else yardline_100
    s["yardline_100"] = np.clip(np.asarray(new_yl, dtype=float), 1, 99)
    s["score_diff"] = -state["score_diff"]
    s["pos_timeouts"] = state["def_timeouts"].to_numpy()
    s["def_timeouts"] = state["pos_timeouts"].to_numpy()
    s["home"] = 1 - state["home"]
    s["spread"] = -state["spread"]
    s["receive_2h"] = np.where(in_first_half(state), 1 - state["receive_2h"], 0)
    s["down"] = 1
    s["ydstogo"] = np.minimum(10, s["yardline_100"])
    return s


def first_down(state: pd.DataFrame, yardline_100) -> pd.DataFrame:
    """Same offense keeps the ball: 1st-and-10 (or goal) at ``yardline_100``."""
    s = state.copy()
    s["yardline_100"] = np.clip(np.asarray(yardline_100, dtype=float), 1, 99)
    s["down"] = 1
    s["ydstogo"] = np.minimum(10, s["yardline_100"])
    return s


def advance_clock(state: pd.DataFrame, seconds: float) -> pd.DataFrame:
    """Run ``seconds`` off the clock (game and half clocks, floored at 0)."""
    s = state.copy()
    gsr = state["game_seconds_remaining"] - seconds
    crossed = (state["game_seconds_remaining"] > HALF) & (gsr <= HALF)
    s["game_seconds_remaining"] = np.where(crossed, HALF, np.maximum(gsr, 0))
    s["half_seconds_remaining"] = np.where(crossed, 0, np.maximum(state["half_seconds_remaining"] - seconds, 0))
    return s


def kickoff_to_opponent(state: pd.DataFrame, points: float, kickoff_yl) -> pd.DataFrame:
    """Offense scores ``points`` then kicks off: opponent's 1st-and-10 at ``kickoff_yl``."""
    s = state.copy()
    s["score_diff"] = state["score_diff"] + points
    return flip_possession(s, kickoff_yl)


def in_first_half(state: pd.DataFrame) -> np.ndarray:
    """True while the first half is still on (including the 0:00 halftime marker)."""
    g, h = state["game_seconds_remaining"], state["half_seconds_remaining"]
    return ((g > HALF) | ((g == HALF) & (h == 0))).to_numpy()


def resolve_halftime(state: pd.DataFrame, kickoff_yl) -> tuple[pd.DataFrame, np.ndarray]:
    """Replace states sitting at the halftime marker by the 2nd-half opening drive.

    :func:`advance_clock` marks "the first half ran out" as ``game_seconds=1800,
    half_seconds=0``. Whatever happened, the team with ``receive_2h`` then gets
    the ball at the kickoff yardline with 3 timeouts each. Returns
    ``(state, swapped)``; ``swapped`` marks rows whose perspective was handed to
    the other team, whose WP must therefore be read as ``1 - wp``.
    """
    s = state.copy()
    at_half = ((s["game_seconds_remaining"] == HALF) & (s["half_seconds_remaining"] == 0)).to_numpy()
    swap = at_half & (s["receive_2h"].to_numpy() == 0)
    if swap.any():
        s.loc[swap, "score_diff"] = -s.loc[swap, "score_diff"]
        s.loc[swap, "home"] = 1 - s.loc[swap, "home"]
        s.loc[swap, "spread"] = -s.loc[swap, "spread"]
    ko = np.broadcast_to(np.asarray(kickoff_yl, dtype=float), (len(s),))
    s.loc[at_half, "yardline_100"] = ko[at_half]
    s.loc[at_half, ["pos_timeouts", "def_timeouts", "down", "ydstogo"]] = [3, 3, 1, 10]
    s.loc[at_half, "half_seconds_remaining"] = HALF
    s.loc[at_half, "receive_2h"] = 0
    return s, swap


def wp_features(state: pd.DataFrame) -> pd.DataFrame:
    """Model features from a state (pure function of the pre-snap state)."""
    X = state[STATE_COLS].astype(float).copy()
    elapsed = 1 - X["game_seconds_remaining"] / GAME
    # Lead scaled up as time runs out (nflfastR-style); monotone in score_diff.
    X["diff_time_ratio"] = X["score_diff"] / np.exp(-4 * elapsed)
    # Pre-game prior fades as the game is played out.
    X["spread_time"] = X["spread"] * (1 - elapsed)
    return X


FEATURES = [*STATE_COLS, "diff_time_ratio", "spread_time"]

#: +1 = WP must not decrease in the feature, -1 = must not increase, 0 = free.
MONOTONE = {
    "score_diff": 1, "game_seconds_remaining": 0, "half_seconds_remaining": 0, "down": -1,
    "ydstogo": -1, "yardline_100": -1, "pos_timeouts": 1, "def_timeouts": -1, "home": 0,
    "spread": 1, "receive_2h": 1, "diff_time_ratio": 1, "spread_time": 1,
}


def terminal_wp(score_diff: np.ndarray) -> np.ndarray:
    """WP once the game clock reads 0:00 (ties count as half a win)."""
    sd = np.asarray(score_diff, dtype=float)
    return np.where(sd > 0, 1.0, np.where(sd < 0, 0.0, 0.5))
