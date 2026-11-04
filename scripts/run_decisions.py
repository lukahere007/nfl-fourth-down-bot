"""Evaluate every 4th down 2014-2025 with the decision engine.

* WP model: leave-one-block-out over 3-season blocks (2014-16, 2017-19,
  2020-22, 2023-25). The model that grades a play never saw that play's game,
  so the game's final result cannot leak into the grade of its own 4th downs.
* Sub-models: fit once on all seasons, plus ``--boot`` game-level cluster
  bootstrap refits that give each call a confidence score.

Writes results/decisions.parquet and results/decision_chart.csv.

Usage: python scripts/run_decisions.py [--boot 100]
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from fourth_down.data import STATE_COLS
from fourth_down.engine import (ACTIONS, bootstrap_confidence, build_curves, recommend,
                                values_from_models)
from fourth_down.submodels import SubModels
from fourth_down.wp_model import MonotoneWP

ROOT = Path(__file__).resolve().parents[1]
DER = ROOT / "data" / "derived"

#: Scenarios for the decision chart: (label, score_diff, game_seconds, half_seconds)
SCENARIOS = [("Tied, 1st quarter (10:00 left)", 0, 3000, 1200),
             ("Down 4, 4th quarter (5:00 left)", -4, 300, 300)]


def load_tables() -> dict[str, pd.DataFrame]:
    return {k: pd.read_parquet(DER / f"{k}.parquet")
            for k in ("wp", "conversion", "field_goal", "punt", "gain", "fourth", "kickoff", "xp")}


def block_of(season: int, first: int = 2014, size: int = 3) -> tuple[int, int]:
    lo = first + (season - first) // size * size
    return lo, lo + size - 1


def held_out_model(wp: pd.DataFrame, season: int) -> MonotoneWP:
    """WP model trained on every season outside ``season``'s 3-season block (cached)."""
    lo, hi = block_of(season)
    path = ROOT / "models" / "held_out" / f"wp_without_{lo}_{hi}.json"
    if path.exists():
        return MonotoneWP.load(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tr = wp[~wp["season"].between(lo, hi)]
    m = MonotoneWP().fit(tr, tr["label_win"])
    m.save(path)
    return m


def decision_chart(model, sm: SubModels, kickoff_yl: float, p_xp: float, season: int) -> pd.DataFrame:
    rows = []
    for label, sd, g, h in SCENARIOS:
        grid = pd.DataFrame([(ytg, yl) for yl in range(1, 100) for ytg in range(1, 16) if ytg <= yl],
                            columns=["ydstogo", "yardline_100"])
        grid = grid.assign(score_diff=sd, game_seconds_remaining=g, half_seconds_remaining=h, down=4,
                           pos_timeouts=3, def_timeouts=3, home=0, spread=0.0, receive_2h=0,
                           season=season, indoor=0, wind=8.0)
        c = build_curves(grid, model, kickoff_yl)
        v = values_from_models(grid, c, sm, p_xp)
        r = recommend(v)
        rows.append(pd.concat([grid[["ydstogo", "yardline_100"]], r,
                               pd.DataFrame(v, columns=[f"wp_{a}" for a in ACTIONS])], axis=1).assign(scenario=label))
    return pd.concat(rows, ignore_index=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--boot", type=int, default=100)
    ap.add_argument("--seasons", type=int, nargs="*")
    a = ap.parse_args()
    T = load_tables()
    ko = T["kickoff"].set_index("season")["kickoff_yardline_100"]
    xp = T["xp"].set_index("season")["xp_rate"]
    sm = SubModels.fit(T)
    rng = np.random.default_rng(2026)
    print(f"fitting {a.boot} bootstrap sub-model sets ...", flush=True)
    boots = [SubModels.fit(T, rng) for _ in range(a.boot)]

    fourth = T["fourth"]
    seasons = a.seasons or sorted(fourth["season"].unique())
    out = []
    for s in seasons:
        plays = fourth[fourth["season"] == s].reset_index(drop=True)
        model = held_out_model(T["wp"], s)
        curves = build_curves(plays, model, ko[s])
        v = values_from_models(plays, curves, sm, xp[s])
        rec = recommend(v)
        conf = bootstrap_confidence(plays, curves, boots, xp[s], rec["best"].to_numpy())
        chosen = plays["decision"].map({act: i for i, act in enumerate(ACTIONS)}).to_numpy()
        res = pd.concat([plays, pd.DataFrame(v, columns=[f"wp_{x}" for x in ACTIONS]), rec, conf], axis=1)
        res["wp_pre"] = model(plays[STATE_COLS])
        res["wp_chosen"] = v[np.arange(len(v)), chosen]
        res["wp_lost"] = res["wp_best"] - res["wp_chosen"]
        out.append(res)
        print(f"{s}: {len(plays):,} 4th downs, bot agrees with coach {np.mean(res.best == res.decision):.1%}", flush=True)
    D = pd.concat(out, ignore_index=True)
    D.to_parquet(ROOT / "results" / "decisions.parquet", index=False, compression="zstd")

    final = MonotoneWP.load(ROOT / "models" / "wp_xgb.json")
    last = int(T["kickoff"]["season"].max())
    decision_chart(final, sm, ko[last], xp[last], last).to_csv(ROOT / "results" / "decision_chart.csv", index=False)
    print("wrote results/decisions.parquet and results/decision_chart.csv")


if __name__ == "__main__":
    main()
