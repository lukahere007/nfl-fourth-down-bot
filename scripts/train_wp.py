"""Walk-forward evaluation of the WP models, then fit the final model.

For each test season 2022-2025 the models are trained only on earlier
seasons. Writes results/wp_metrics.csv, results/wp_calibration.csv,
results/wp_monotonicity.csv and models/wp_xgb.json (fit on all seasons).

Usage: python scripts/train_wp.py [--test-seasons 2022 2023 2024 2025]
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from fourth_down.wp_model import (LogisticWP, MonotoneWP, brier, calibration_table, log_loss,
                                  monotonicity_violations)

ROOT = Path(__file__).resolve().parents[1]
PHASES = {1: "Q1", 2: "Q2", 3: "Q3", 4: "Q4 (>5 min)", 5: "Last 5 min"}


def phase(df: pd.DataFrame) -> pd.Series:
    g = df["game_seconds_remaining"]
    q = np.select([g > 2700, g > 1800, g > 900, g > 300], [1, 2, 3, 4], 5)
    return pd.Series(q, index=df.index).map(PHASES)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--test-seasons", type=int, nargs="+", default=[2022, 2023, 2024, 2025])
    a = ap.parse_args()
    wp = pd.read_parquet(ROOT / "data" / "derived" / "wp.parquet")
    preds = []
    mono_rows = []
    for s in a.test_seasons:
        tr, te = wp[wp["season"] < s], wp[wp["season"] == s].copy()
        print(f"season {s}: train {len(tr):,} snaps ({tr.season.min()}-{tr.season.max()}), test {len(te):,}", flush=True)
        mono = MonotoneWP().fit(tr, tr["label_win"])
        te["p_xgb_mono"] = mono.predict(te)
        te["p_logistic"] = LogisticWP().fit(tr, tr["label_win"]).predict(te)
        if s == a.test_seasons[0]:  # one unconstrained fit to show what the constraints buy
            free = MonotoneWP(monotone=False).fit(tr, tr["label_win"])
            te["p_xgb_free"] = free.predict(te)
            probe = te.sample(20000, random_state=0)
            for name, m in [("xgb_mono", mono), ("xgb_free", free)]:
                for col, step in [("score_diff", 1), ("yardline_100", -1), ("spread", 0.5), ("pos_timeouts", 1)]:
                    ok = probe[col] + step <= {"pos_timeouts": 3}.get(col, 1e9)
                    mono_rows.append({"model": name, "feature": col, "step": step,
                                      "violation_rate": monotonicity_violations(m, probe[ok], col, step, 1)})
        preds.append(te)
    P = pd.concat(preds)
    P["phase"] = phase(P)
    models = {"xgb_mono": "p_xgb_mono", "logistic": "p_logistic", "nflverse_wp": "wp", "nflverse_vegas_wp": "vegas_wp"}
    rows = []
    for name, col in models.items():
        rows.append({"model": name, "season": "all", "log_loss": log_loss(P.label_win, P[col]),
                     "brier": brier(P.label_win, P[col]), "n": len(P)})
        for s, g in P.groupby("season"):
            rows.append({"model": name, "season": str(s), "log_loss": log_loss(g.label_win, g[col]),
                         "brier": brier(g.label_win, g[col]), "n": len(g)})
    f = P[P["p_xgb_free"].notna()]
    rows.append({"model": "xgb_unconstrained", "season": str(a.test_seasons[0]),
                 "log_loss": log_loss(f.label_win, f.p_xgb_free), "brier": brier(f.label_win, f.p_xgb_free), "n": len(f)})
    met = pd.DataFrame(rows)
    met.to_csv(ROOT / "results" / "wp_metrics.csv", index=False)
    print(met[met.season == "all"].to_string(index=False))

    cal = []
    for ph, g in P.groupby("phase"):
        for name in ("xgb_mono", "logistic"):
            t = calibration_table(g.label_win.to_numpy(), g[models[name]].to_numpy())
            cal.append(t.assign(phase=ph, model=name, log_loss=log_loss(g.label_win, g[models[name]])))
    pd.concat(cal).to_csv(ROOT / "results" / "wp_calibration.csv", index=False)
    pd.DataFrame(mono_rows).to_csv(ROOT / "results" / "wp_monotonicity.csv", index=False)
    print(pd.DataFrame(mono_rows).to_string(index=False))

    final = MonotoneWP().fit(wp, wp["label_win"])
    (ROOT / "models").mkdir(exist_ok=True)
    final.save(ROOT / "models" / "wp_xgb.json")
    print("saved models/wp_xgb.json")


if __name__ == "__main__":
    main()
