"""Ask the bot about one 4th-down situation.

Example (4th-and-2 at the opponent's 38, down 3, 6:00 left in the 4th):
    python scripts/decide.py --ydstogo 2 --yardline 38 --score-diff -3 --minutes-left 6

Uses the committed WP model (models/wp_xgb.json) and sub-models fit on the
committed derived tables; ``--boot`` adds a bootstrap confidence.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from fourth_down.engine import ACTIONS, bootstrap_confidence, build_curves, recommend, values_from_models
from fourth_down.submodels import SubModels
from fourth_down.wp_model import MonotoneWP

ROOT = Path(__file__).resolve().parents[1]
DER = ROOT / "data" / "derived"


def situation(a) -> pd.DataFrame:
    gsr = a.minutes_left * 60
    hsr = gsr - 1800 if gsr > 1800 else gsr
    return pd.DataFrame([dict(score_diff=a.score_diff, game_seconds_remaining=gsr, half_seconds_remaining=hsr,
                              down=4, ydstogo=a.ydstogo, yardline_100=a.yardline, pos_timeouts=a.timeouts,
                              def_timeouts=a.opp_timeouts, home=int(a.home), spread=a.spread,
                              receive_2h=int(a.receive_2h and gsr > 1800), season=a.season,
                              indoor=int(a.indoor), wind=a.wind)])


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--ydstogo", type=float, required=True)
    p.add_argument("--yardline", type=float, required=True, help="yards from the opponent's end zone")
    p.add_argument("--score-diff", type=float, default=0, help="offense minus defense")
    p.add_argument("--minutes-left", type=float, default=30, help="minutes left in regulation")
    p.add_argument("--timeouts", type=int, default=3)
    p.add_argument("--opp-timeouts", type=int, default=3)
    p.add_argument("--spread", type=float, default=0, help="pre-game expected margin for the offense")
    p.add_argument("--home", action="store_true")
    p.add_argument("--receive-2h", action="store_true")
    p.add_argument("--indoor", action="store_true")
    p.add_argument("--wind", type=float, default=8)
    p.add_argument("--season", type=int, default=2025)
    p.add_argument("--boot", type=int, default=0)
    a = p.parse_args()

    T = {k: pd.read_parquet(DER / f"{k}.parquet") for k in ("conversion", "field_goal", "punt", "gain", "kickoff", "xp")}
    ko = T["kickoff"].set_index("season")["kickoff_yardline_100"][a.season]
    xp = T["xp"].set_index("season")["xp_rate"][a.season]
    wp = MonotoneWP.load(ROOT / "models" / "wp_xgb.json")
    sm = SubModels.fit(T)
    s = situation(a)
    curves = build_curves(s, wp, ko)
    v = values_from_models(s, curves, sm, xp)
    r = recommend(v).iloc[0]
    print(f"4th-and-{a.ydstogo:g} at {a.yardline:g} yards from goal, score diff {a.score_diff:+g}, "
          f"{a.minutes_left:g} min left")
    print(f"  WP now (before the snap): {wp(s)[0]:.1%}")
    for act, val in zip(ACTIONS, v[0]):
        print(f"  {act:5s} {val:6.1%}")
    msg = f"  -> {r.best.upper()} (by {r.gap * 100:.1f} WP points)"
    if a.boot:
        rng = np.random.default_rng(0)
        boots = [SubModels.fit(T, rng) for _ in range(a.boot)]
        c = bootstrap_confidence(s, curves, boots, xp, np.array([r.best])).iloc[0]
        msg += f", bootstrap confidence {c.confidence:.0%}"
    print(msg)


if __name__ == "__main__":
    main()
