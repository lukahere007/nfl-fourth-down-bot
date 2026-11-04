"""Win-probability models: monotone gradient boosting and a logistic baseline."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer

from .state import FEATURES, MONOTONE, wp_features

DEFAULT_PARAMS = dict(
    eta=0.1, max_depth=6, min_child_weight=200, subsample=0.8, colsample_bytree=0.8,
    reg_lambda=5.0, tree_method="hist", objective="binary:logistic", eval_metric="logloss",
    nthread=-1,
)
N_ROUNDS = 200


@dataclass
class MonotoneWP:
    """XGBoost WP model with football-logic monotone constraints.

    The constraints (see :data:`fourth_down.state.MONOTONE`) guarantee, e.g.,
    that adding a point to the offense's lead can never lower its WP, and that
    being closer to the end zone can never hurt. Without them a tree model can
    learn small non-monotone wiggles from noise, and a decision engine that
    compares WP between neighbouring states turns those wiggles into bad calls.
    """

    params: dict = field(default_factory=lambda: dict(DEFAULT_PARAMS))
    monotone: bool = True
    n_rounds: int = N_ROUNDS
    model: xgb.Booster | None = None

    def fit(self, states: pd.DataFrame, y: np.ndarray, seed: int = 0) -> "MonotoneWP":
        """Fit on pre-snap states; ``y`` is 1 win / 0 loss / 0.5 tie (soft label)."""
        X = wp_features(states)[FEATURES]
        p = dict(self.params, seed=seed)
        if self.monotone:
            p["monotone_constraints"] = "(" + ",".join(str(MONOTONE[f]) for f in FEATURES) + ")"
        dtrain = xgb.DMatrix(X, label=np.asarray(y, dtype=float))
        self.model = xgb.train(p, dtrain, num_boost_round=self.n_rounds)
        return self

    def predict(self, states: pd.DataFrame) -> np.ndarray:
        X = wp_features(states)[FEATURES]
        return self.model.predict(xgb.DMatrix(X))

    __call__ = predict

    def save(self, path: Path) -> None:
        self.model.save_model(str(path))

    @classmethod
    def load(cls, path: Path) -> "MonotoneWP":
        b = xgb.Booster()
        b.load_model(str(path))
        return cls(model=b)


class LogisticWP:
    """Baseline: logistic regression on hand-built features (no interactions learned)."""

    num = ["score_diff", "diff_time_ratio", "spread", "spread_time", "yardline_100", "ydstogo",
           "game_seconds_remaining", "half_seconds_remaining", "pos_timeouts", "def_timeouts",
           "home", "receive_2h"]

    def fit(self, states: pd.DataFrame, y: np.ndarray) -> "LogisticWP":
        X = wp_features(states)
        ct = ColumnTransformer([("num", StandardScaler(), self.num),
                                ("down", OneHotEncoder(drop="first"), ["down"])])
        self.pipe = make_pipeline(ct, LogisticRegression(C=1.0, max_iter=2000))
        yy = np.asarray(y)
        keep = yy != 0.5  # sklearn needs hard labels; ties are <0.4% of snaps
        self.pipe.fit(X[keep], yy[keep].astype(int))
        return self

    def predict(self, states: pd.DataFrame) -> np.ndarray:
        return self.pipe.predict_proba(wp_features(states))[:, 1]


def log_loss(y: np.ndarray, p: np.ndarray) -> float:
    """Mean binary log loss (accepts 0.5 labels for ties)."""
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def brier(y: np.ndarray, p: np.ndarray) -> float:
    return float(np.mean((np.asarray(p) - np.asarray(y)) ** 2))


def calibration_table(y: np.ndarray, p: np.ndarray, bins: int = 10) -> pd.DataFrame:
    """Mean predicted vs observed win rate in equal-width probability bins."""
    b = np.clip((np.asarray(p) * bins).astype(int), 0, bins - 1)
    df = pd.DataFrame({"bin": b, "p": p, "y": y})
    return df.groupby("bin").agg(pred=("p", "mean"), obs=("y", "mean"), n=("y", "size")).reset_index()


def monotonicity_violations(wp_fn, states: pd.DataFrame, col: str, step: float, direction: int) -> float:
    """Share of states where moving ``col`` by ``step`` moves WP against ``direction``."""
    a = wp_fn(states)
    s2 = states.copy()
    s2[col] = s2[col] + step
    b = wp_fn(s2)
    return float(np.mean(direction * (b - a) < -1e-9))
