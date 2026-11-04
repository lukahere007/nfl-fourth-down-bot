"""Small, dependency-free smoothers used by the outcome sub-models.

:class:`ShapeConstrainedLogit` is a logistic GAM whose smooth terms are
piecewise-linear "ramp" splines. Each ramp basis function rises from 0 to 1
across one knot interval, so its coefficient is the change of the logit over
that interval. That makes shape constraints trivial: a term is monotone
decreasing exactly when all of its coefficients are <= 0, which L-BFGS-B
enforces with simple bounds. A second-difference penalty on neighbouring
slopes (a P-spline idea) keeps the curve smooth where data are thin.

:func:`kernel_quantiles` + isotonic regression gives smooth conditional
quantiles of a continuous outcome (used for net punt results).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit
from sklearn.isotonic import IsotonicRegression


def ramp_basis(x: np.ndarray, knots: np.ndarray) -> np.ndarray:
    """Columns ``clip((x - k_j) / (k_{j+1} - k_j), 0, 1)`` for consecutive knots.

    Each column rises from 0 to 1 across one knot interval, so its coefficient
    is the change in the logit across that interval (slope x width).
    """
    x = np.asarray(x, dtype=float)[:, None]
    lo, hi = knots[:-1][None, :], knots[1:][None, :]
    return np.clip((x - lo) / (hi - lo), 0, 1)


@dataclass
class Smooth:
    """A smooth term: column name, knots, and shape (``"dec"``, ``"inc"`` or ``None``)."""

    col: str
    knots: list[float]
    shape: str | None = None


@dataclass
class ShapeConstrainedLogit:
    """Penalised logistic GAM with monotone piecewise-linear smooth terms."""

    smooths: list[Smooth]
    linear: list[str] = field(default_factory=list)
    lam: float = 1.0
    coef_: np.ndarray | None = None

    def _design(self, df) -> np.ndarray:
        blocks = [ramp_basis(df[s.col].to_numpy(), np.asarray(s.knots, float)) for s in self.smooths]
        if self.linear:
            z = df[self.linear].to_numpy(dtype=float)
            blocks.append((z - self.mu_) / self.sd_)
        return np.hstack([np.ones((len(df), 1)), *blocks])

    def _bounds(self) -> list[tuple]:
        b = [(None, None)]
        for s in self.smooths:
            m = len(s.knots) - 1
            b += [{"dec": (None, 0.0), "inc": (0.0, None)}.get(s.shape, (None, None))] * m
        return b + [(None, None)] * len(self.linear)

    def _penalty_matrix(self, p: int) -> np.ndarray:
        D = []
        j = 1
        for s in self.smooths:
            width = np.diff(np.asarray(s.knots, float))
            m = len(width)
            for i in range(m - 1):  # difference of neighbouring slopes (coef / width)
                row = np.zeros(p)
                row[j + i], row[j + i + 1] = -1.0 / width[i], 1.0 / width[i + 1]
                D.append(row)
            j += m
        return np.array(D) if D else np.zeros((0, p))

    def fit(self, df, y, weights=None) -> "ShapeConstrainedLogit":
        if self.linear:
            z = df[self.linear].to_numpy(dtype=float)
            self.mu_, self.sd_ = z.mean(0), z.std(0) + 1e-9
        X = self._design(df)
        y = np.asarray(y, dtype=float)
        w = np.ones(len(y)) if weights is None else np.asarray(weights, float)
        w = w / w.sum()
        D = self._penalty_matrix(X.shape[1])
        P = D.T @ D
        lam = self.lam / len(y) * 100  # penalty strength roughly independent of n

        def f(beta):
            eta = X @ beta
            nll = np.sum(w * (np.logaddexp(0, eta) - y * eta))
            g = X.T @ (w * (expit(eta) - y))
            return nll + lam * beta @ P @ beta, g + 2 * lam * P @ beta

        x0 = np.zeros(X.shape[1])
        x0[0] = np.log((y.mean() + 1e-3) / (1 - y.mean() + 1e-3))
        res = minimize(f, x0, jac=True, method="L-BFGS-B", bounds=self._bounds(), options={"maxiter": 2000})
        self.coef_ = res.x
        self.n_iter_ = res.nit
        return self

    def predict(self, df) -> np.ndarray:
        return expit(self._design(df) @ self.coef_)


def kernel_quantiles(x: np.ndarray, y: np.ndarray, grid: np.ndarray, qs: np.ndarray,
                     bandwidth: float, decreasing: bool = True) -> np.ndarray:
    """Gaussian-kernel weighted quantiles of ``y`` at each ``grid`` point of ``x``.

    Each quantile curve is then made monotone in ``x`` with isotonic regression
    (``decreasing=True`` for punts: punting from deeper gives the receiver a
    better spot). Returns an array of shape ``(len(grid), len(qs))``.
    """
    x, y = np.asarray(x, float), np.asarray(y, float)
    order = np.argsort(y)
    x, y = x[order], y[order]
    out = np.empty((len(grid), len(qs)))
    for i, g in enumerate(grid):
        w = np.exp(-0.5 * ((x - g) / bandwidth) ** 2)
        cw = np.cumsum(w) / w.sum()
        out[i] = np.interp(qs, cw, y)
    iso = IsotonicRegression(increasing=not decreasing)
    for j in range(len(qs)):
        out[:, j] = iso.fit_transform(grid, out[:, j])
    return np.sort(out, axis=1)
