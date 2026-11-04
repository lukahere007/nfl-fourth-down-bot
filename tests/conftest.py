import numpy as np
import pandas as pd
import pytest
from scipy.stats import norm


def make_state(**kw) -> pd.DataFrame:
    """One-row state with sensible defaults (tie game, start of Q2, 3 timeouts each)."""
    base = dict(score_diff=0.0, game_seconds_remaining=2700.0, half_seconds_remaining=900.0,
                down=4.0, ydstogo=4.0, yardline_100=40.0, pos_timeouts=3.0, def_timeouts=3.0,
                home=1.0, spread=0.0, receive_2h=0.0)
    base.update(kw)
    return pd.DataFrame([base])


def toy_wp(s: pd.DataFrame) -> np.ndarray:
    """A transparent analytic WP: expected margin / remaining-time volatility.

    Field position is worth ~6 points at the goal line and about -1 deep in
    your own end, the spread fades with time, and the volatility of the final
    margin shrinks like sqrt(time left). Monotone in score and field position.
    """
    t = s["game_seconds_remaining"].to_numpy() / 3600
    ep = 6.0 - 0.075 * s["yardline_100"].to_numpy()
    margin = s["score_diff"].to_numpy() + ep + s["spread"].to_numpy() * t
    return norm.cdf(margin / (13.5 * np.sqrt(t) + 0.5))


@pytest.fixture
def state():
    return make_state
