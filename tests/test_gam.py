import numpy as np
import pandas as pd

from fourth_down.gam import ShapeConstrainedLogit, Smooth, kernel_quantiles, ramp_basis


def test_ramp_basis_rows_sum_to_position():
    B = ramp_basis(np.array([0.0, 1.5, 10.0]), np.array([0.0, 1.0, 2.0, 4.0]))
    assert np.allclose(B, [[0, 0, 0], [1, 0.5, 0], [1, 1, 1]])


def test_monotone_logit_is_monotone_and_recovers_curve():
    rng = np.random.default_rng(0)
    x = rng.uniform(1, 20, 20000)
    p_true = 1 / (1 + np.exp(-(1.5 - 0.15 * x)))
    y = rng.random(len(x)) < p_true
    # Add a few contrary outcomes at long range to tempt an unconstrained fit.
    y[(x > 18) & (rng.random(len(x)) < 0.2)] = True
    df = pd.DataFrame({"x": x})
    m = ShapeConstrainedLogit([Smooth("x", [1, 3, 6, 10, 15, 20], "dec")]).fit(df, y)
    grid = pd.DataFrame({"x": np.linspace(1, 20, 60)})
    p = m.predict(grid)
    assert np.all(np.diff(p) <= 1e-12)
    assert np.max(np.abs(p[:40] - 1 / (1 + np.exp(-(1.5 - 0.15 * grid.x[:40]))))) < 0.05


def test_kernel_quantiles_monotone_and_ordered():
    rng = np.random.default_rng(1)
    x = rng.uniform(30, 99, 5000)
    y = np.clip(100 - (x - rng.normal(42, 8, len(x))), 1, 99)
    q = kernel_quantiles(x, y, np.arange(30, 100), np.array([0.1, 0.5, 0.9]), 4.0, decreasing=True)
    assert np.all(np.diff(q, axis=0) <= 1e-9)    # deeper punts -> better start for receiver
    assert np.all(np.diff(q, axis=1) >= 0)       # quantiles ordered
    assert abs(q[np.arange(30, 100) == 70, 1][0] - 72) < 2
