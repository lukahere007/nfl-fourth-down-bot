import numpy as np
import pytest

from fourth_down.engine import Curves, action_values, build_curves, recommend
from conftest import make_state, toy_wp


def flat_curves(own=0.6, opp=0.3, w3=0.5, w6=0.7, w7=0.8, n=1):
    return Curves(own=np.full((n, 99), own), opp=np.full((n, 99), opp),
                  score=np.tile([w3, w6, w7], (n, 1)))


def test_expected_wp_arithmetic_toy_case():
    c = flat_curves()
    v = action_values([40], [4], c, p_convert=np.array([0.5]), extra_yards=np.zeros((1, 3)),
                      p_fg=np.array([0.8]), punt_opp_yl=np.full((1, 5), 80.0), p_xp=0.9)
    go, fg, punt = v[0]
    assert go == pytest.approx(0.5 * 0.6 + 0.5 * 0.3)        # 0.45
    assert fg == pytest.approx(0.8 * 0.5 + 0.2 * 0.3)        # 0.46
    assert punt == pytest.approx(0.3)


def test_goal_to_go_success_is_a_touchdown():
    c = flat_curves()
    v = action_values([2], [2], c, np.array([1.0]), np.zeros((1, 3)), np.array([0.0]),
                      np.full((1, 3), 80.0), p_xp=0.9)
    assert v[0, 0] == pytest.approx(0.9 * 0.8 + 0.1 * 0.7)


def test_outcome_lookup_uses_the_right_yardline():
    own = np.tile(np.linspace(1.0, 0.0, 99), (1, 1))     # own[y] falls with distance
    opp = np.tile(np.arange(1, 100) / 100.0, (1, 1))     # opp[y] = y / 100
    c = Curves(own, opp, np.array([[0.5, 0.7, 0.8]]))
    v = action_values([50], [5], c, np.array([0.0]), np.zeros((1, 1)), np.array([0.0]),
                      np.array([[70.0, 90.0]]), p_xp=1.0)
    assert v[0, 0] == pytest.approx(0.50)                 # failed: opponent at its own 50
    assert v[0, 1] == pytest.approx(0.43)                 # missed FG from 50: spot of kick, opp at 43
    assert v[0, 2] == pytest.approx(0.80)                 # punt: mean of 0.70 and 0.90


def _decide(state):
    c = build_curves(state, toy_wp, kickoff_yl=75)
    yl, ytg = state["yardline_100"].to_numpy(), state["ydstogo"].to_numpy()
    p_conv = np.clip(0.72 * np.exp(-0.12 * (ytg - 1)), 0, 1)
    dist = yl + 18
    p_fg = np.where(dist > 70, 0.0, 1 / (1 + np.exp(-(6.0 - 0.11 * dist))))
    net = np.linspace(30, 52, 6)
    punt = np.where(yl[:, None] - net[None, :] <= 0, 80.0, 100 - (yl[:, None] - net[None, :]))
    v = action_values(yl, ytg, c, p_conv, np.zeros((1, 4)), p_fg, punt, p_xp=0.94)
    return recommend(v).iloc[0], v[0]


def test_engine_goes_for_it_4th_and_1_at_the_1_down_4_late():
    rec, v = _decide(make_state(score_diff=-4, game_seconds_remaining=120, half_seconds_remaining=120,
                                ydstogo=1, yardline_100=1))
    assert rec.best == "go"
    assert v[0] > v[1] + 0.1                                # a FG still leaves them losing


def test_engine_punts_4th_and_20_at_own_10_early():
    rec, v = _decide(make_state(score_diff=0, game_seconds_remaining=3300, half_seconds_remaining=1500,
                                ydstogo=20, yardline_100=90))
    assert rec.best == "punt"
    assert v[2] > max(v[0], v[1]) + 0.02


def test_engine_handles_end_of_game():
    # Up 2, 4th down with 3 seconds left: the game ends on the snap, every call wins.
    rec, v = _decide(make_state(score_diff=2, game_seconds_remaining=3, half_seconds_remaining=3,
                                ydstogo=8, yardline_100=60))
    assert np.allclose(v, 1.0)


def test_recommend_reports_gap():
    r = recommend(np.array([[0.40, 0.45, 0.30]])).iloc[0]
    assert r.best == "fg" and r.gap == pytest.approx(0.05)
