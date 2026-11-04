from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from fourth_down.wp_model import MonotoneWP, monotonicity_violations
from conftest import make_state, toy_wp

ROOT = Path(__file__).resolve().parents[1]
MODEL = ROOT / "models" / "wp_xgb.json"


def synthetic_states(n, rng):
    g = rng.uniform(0, 3600, n)
    return pd.DataFrame({
        "score_diff": rng.integers(-21, 22, n).astype(float), "game_seconds_remaining": g,
        "half_seconds_remaining": np.where(g > 1800, g - 1800, g), "down": rng.integers(1, 5, n).astype(float),
        "ydstogo": rng.integers(1, 15, n).astype(float), "yardline_100": rng.integers(1, 100, n).astype(float),
        "pos_timeouts": rng.integers(0, 4, n).astype(float), "def_timeouts": rng.integers(0, 4, n).astype(float),
        "home": rng.integers(0, 2, n).astype(float), "spread": rng.normal(0, 5, n).round(1),
        "receive_2h": np.where(g > 1800, rng.integers(0, 2, n), 0).astype(float),
    })


def test_monotone_constraints_hold_on_noisy_synthetic_data():
    rng = np.random.default_rng(0)
    s = synthetic_states(20000, rng)
    y = (rng.random(len(s)) < toy_wp(s)).astype(float)
    m = MonotoneWP(n_rounds=60).fit(s, y)
    probe = synthetic_states(3000, np.random.default_rng(1))
    assert monotonicity_violations(m, probe, "score_diff", 1, +1) == 0
    assert monotonicity_violations(m, probe, "yardline_100", -5, +1) == 0
    assert monotonicity_violations(m, probe, "spread", 1, +1) == 0


@pytest.mark.skipif(not MODEL.exists(), reason="trained model not built yet")
def test_committed_model_is_monotone_and_sane():
    m = MonotoneWP.load(MODEL)
    probe = synthetic_states(5000, np.random.default_rng(2))
    assert monotonicity_violations(m, probe, "score_diff", 1, +1) == 0
    assert monotonicity_violations(m, probe, "yardline_100", -1, +1) == 0
    up14 = m(make_state(score_diff=14, game_seconds_remaining=120, half_seconds_remaining=120, down=1))
    assert up14[0] > 0.97
    kickoff = m(make_state(game_seconds_remaining=3600, half_seconds_remaining=1800, down=1, ydstogo=10,
                           yardline_100=75, receive_2h=0))
    assert 0.4 < kickoff[0] < 0.6
