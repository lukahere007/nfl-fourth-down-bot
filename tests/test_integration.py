"""End-to-end checks on the committed model and derived data (skipped if absent)."""
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from fourth_down.engine import build_curves, recommend, values_from_models
from fourth_down.submodels import SubModels
from fourth_down.wp_model import MonotoneWP
from conftest import make_state

ROOT = Path(__file__).resolve().parents[1]
DER = ROOT / "data" / "derived"
MODEL = ROOT / "models" / "wp_xgb.json"
pytestmark = pytest.mark.skipif(not (MODEL.exists() and (DER / "conversion.parquet").exists()),
                                reason="model / derived data not built")


@pytest.fixture(scope="module")
def bot():
    T = {k: pd.read_parquet(DER / f"{k}.parquet") for k in ("conversion", "field_goal", "punt", "gain")}
    return MonotoneWP.load(MODEL), SubModels.fit(T)


def call(bot, **kw):
    wp, sm = bot
    s = make_state(**kw).assign(season=2025, indoor=0, wind=8.0)
    v = values_from_models(s, build_curves(s, wp, 69), sm, 0.95)
    return recommend(v).iloc[0], v[0]


def test_real_bot_goes_4th_and_1_at_the_1_down_4_late(bot):
    rec, _ = call(bot, score_diff=-4, game_seconds_remaining=120, half_seconds_remaining=120,
                  ydstogo=1, yardline_100=1)
    assert rec.best == "go"


def test_real_bot_punts_4th_and_20_at_own_10_early(bot):
    rec, _ = call(bot, game_seconds_remaining=3300, half_seconds_remaining=1500, ydstogo=20, yardline_100=90)
    assert rec.best == "punt"


def test_real_bot_kicks_chip_shot_to_win_at_the_end(bot):
    # Tied, 4th-and-8 at the opponent's 10 with 5 seconds left: kick the 28-yarder.
    rec, v = call(bot, score_diff=0, game_seconds_remaining=5, half_seconds_remaining=5,
                  ydstogo=8, yardline_100=10)
    assert rec.best == "fg" and v[1] > 0.9


def test_sub_models_are_monotone(bot):
    _, sm = bot
    g = pd.DataFrame({"ydstogo": np.arange(1, 16), "yardline_100": 50, "spread": 0.0, "season": 2025})
    assert np.all(np.diff(sm.conversion.predict(g)) <= 0)
    k = pd.DataFrame({"yardline_100": np.arange(1, 50), "season": 2025, "indoor": 0, "wind": 8.0})
    assert np.all(np.diff(sm.field_goal.predict(k)) <= 0)
