"""Outcome sub-models: conversion, field goals, punts, and yards on success.

Each one is fit on plays from *all* teams, so they describe a league-average
offense/kicker/punter in that season (team strength enters only via the
pre-game spread in the conversion model). ``SubModels.fit(..., rng=...)``
refits everything on a game-level cluster bootstrap sample, which is how the
decision engine measures how sure it is about a call.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .gam import ShapeConstrainedLogit, Smooth, kernel_quantiles

MAX_FG = 70  # longest attempt in the data; longer kicks are treated as impossible
PUNT_Q = np.linspace(0.05, 0.95, 10)
GAIN_Q = np.linspace(0.05, 0.95, 10)


def cluster_resample(df: pd.DataFrame, rng: np.random.Generator, col: str = "game_id") -> pd.DataFrame:
    """Resample whole games with replacement (plays in a game are correlated)."""
    games = df[col].unique()
    pick = rng.choice(games, size=len(games), replace=True)
    counts = pd.Series(pick).value_counts()
    w = df[col].map(counts).fillna(0).astype(int).to_numpy()
    return df.loc[np.repeat(df.index.to_numpy(), w)]


class ConversionModel:
    """P(convert | distance, field position, down, spread, season)."""

    def fit(self, conv: pd.DataFrame) -> "ConversionModel":
        d = conv.assign(is_4th=(conv["down"] == 4).astype(float))
        self.m = ShapeConstrainedLogit(
            smooths=[Smooth("ydstogo", [1, 2, 3, 4, 5, 7, 10, 15, 25], "dec"),
                     Smooth("yardline_100", [1, 2, 3, 5, 10, 20, 40, 60, 80, 99], None)],
            linear=["is_4th", "spread", "season"], lam=1.0,
        ).fit(d, d["converted"])
        return self

    def predict(self, plays: pd.DataFrame) -> np.ndarray:
        return self.m.predict(plays.assign(is_4th=1.0))


class FieldGoalModel:
    """P(make | kick distance, season, indoor, wind)."""

    def fit(self, fg: pd.DataFrame) -> "FieldGoalModel":
        self.m = ShapeConstrainedLogit(
            smooths=[Smooth("kick_distance", list(range(18, MAX_FG, 4)) + [MAX_FG], "dec")],
            linear=["season", "indoor", "wind"], lam=1.0,
        ).fit(fg, fg["made"])
        return self

    def predict(self, plays: pd.DataFrame) -> np.ndarray:
        dist = plays["yardline_100"] + 18
        p = self.m.predict(plays.assign(kick_distance=dist))
        return np.where(dist > MAX_FG, 0.0, p)


class PuntModel:
    """Quantiles of the receiving team's starting yardline after a punt."""

    def fit(self, punt: pd.DataFrame, bandwidth: float = 4.0) -> "PuntModel":
        self.grid = np.arange(1, 100)
        x = punt["yardline_100"].to_numpy()
        y = punt["opp_yardline_100"].to_numpy()
        self.q = kernel_quantiles(x, y, self.grid, PUNT_Q, bandwidth, decreasing=True)
        # Below the shortest punt in the data, reuse that row (touchback-heavy).
        lo = int(x.min())
        self.q[: lo - 1] = self.q[lo - 1]
        return self

    def predict(self, plays: pd.DataFrame) -> np.ndarray:
        """Array (n_plays, 10) of equally likely opponent starting yardlines."""
        i = np.clip(plays["yardline_100"].round().astype(int).to_numpy(), 1, 99) - 1
        return self.q[i]


class GainModel:
    """Quantiles of yards beyond the line to gain on a successful conversion."""

    buckets = [(1, 2), (3, 5), (6, 99)]

    def fit(self, gain: pd.DataFrame) -> "GainModel":
        self.q = {}
        for lo, hi in self.buckets:
            g = gain[gain["ydstogo"].between(lo, hi)]["extra"].to_numpy()
            self.q[(lo, hi)] = np.quantile(g, GAIN_Q)
        return self

    def predict(self, plays: pd.DataFrame) -> np.ndarray:
        """Array (n_plays, 10) of equally likely extra yards."""
        ytg = plays["ydstogo"].to_numpy()
        out = np.zeros((len(plays), len(GAIN_Q)))
        for (lo, hi), q in self.q.items():
            out[(ytg >= lo) & (ytg <= hi)] = q
        return out


@dataclass
class SubModels:
    conversion: ConversionModel
    field_goal: FieldGoalModel
    punt: PuntModel
    gain: GainModel

    @classmethod
    def fit(cls, tables: dict[str, pd.DataFrame], rng: np.random.Generator | None = None) -> "SubModels":
        t = {k: tables[k] for k in ("conversion", "field_goal", "punt", "gain")}
        if rng is not None:
            t = {k: cluster_resample(v, rng) for k, v in t.items()}
        return cls(ConversionModel().fit(t["conversion"]), FieldGoalModel().fit(t["field_goal"]),
                   PuntModel().fit(t["punt"]), GainModel().fit(t["gain"]))
