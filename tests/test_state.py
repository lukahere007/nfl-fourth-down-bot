import numpy as np
import pandas as pd

from fourth_down.state import (advance_clock, flip_possession, kickoff_to_opponent,
                               resolve_halftime, wp_features)
from conftest import make_state


def test_flip_swaps_perspective():
    s = make_state(score_diff=7, yardline_100=30, pos_timeouts=2, def_timeouts=1,
                   home=1, spread=3.5, receive_2h=1)
    f = flip_possession(s).iloc[0]
    assert f.score_diff == -7
    assert f.yardline_100 == 70          # same spot, other team's distance to goal
    assert (f.pos_timeouts, f.def_timeouts) == (1, 2)
    assert f.home == 0 and f.spread == -3.5
    assert f.receive_2h == 0              # first half: the other team gets the 2H kick
    assert (f.down, f.ydstogo) == (1, 10)


def test_flip_twice_is_identity_on_game_situation():
    s = make_state(score_diff=-3, yardline_100=63, pos_timeouts=1, spread=-2, receive_2h=1)
    ff = flip_possession(flip_possession(s))
    cols = ["score_diff", "yardline_100", "pos_timeouts", "def_timeouts", "home", "spread", "receive_2h"]
    pd.testing.assert_frame_equal(ff[cols], s[cols], check_dtype=False)


def test_flip_to_explicit_spot_and_goal_to_go():
    s = make_state(yardline_100=50)
    f = flip_possession(s, yardline_100=np.array([4.0])).iloc[0]
    assert f.yardline_100 == 4 and f.ydstogo == 4   # 1st-and-goal from the 4


def test_receive_2h_is_zero_in_second_half():
    s = make_state(game_seconds_remaining=1000, half_seconds_remaining=1000, receive_2h=0)
    assert flip_possession(s).iloc[0].receive_2h == 0


def test_kickoff_after_score():
    s = make_state(score_diff=-4, yardline_100=1)
    k = kickoff_to_opponent(s, 7, 75).iloc[0]
    assert k.score_diff == -3                # opponent now trails by 3
    assert k.yardline_100 == 75


def test_advance_clock_marks_halftime_and_resolve_gives_ball_to_receiver():
    s = make_state(game_seconds_remaining=1803, half_seconds_remaining=3, receive_2h=0, score_diff=3)
    a = advance_clock(s, 6).iloc[0]
    assert (a.game_seconds_remaining, a.half_seconds_remaining) == (1800, 0)
    r, swapped = resolve_halftime(advance_clock(s, 6), 75)
    row = r.iloc[0]
    assert swapped[0]                         # other team receives the 2nd half
    assert row.score_diff == -3 and row.yardline_100 == 75
    assert (row.half_seconds_remaining, row.pos_timeouts, row.def_timeouts) == (1800, 3, 3)


def test_features_are_pre_snap_only():
    X = wp_features(make_state())
    assert set(X.columns) >= {"diff_time_ratio", "spread_time"}
    assert not any("label" in c or c in {"wp", "vegas_wp", "result"} for c in X.columns)
