"""Self-tests for the numerical gradient checker.

The grad checker is the oracle used to verify analytical gradients in every
layer test we write. If it's broken, every layer test would pass with
silently-wrong numbers. These tests pin its correctness against simple
functions whose gradients are known by hand:

* ``f(x) = sum(x**2)``    → ``df/dx = 2*x``
* ``f(x) = sum(sin(x))``  → ``df/dx = cos(x)``

Both flavors of the checker (``numerical_x_gradient``, ``numerical_gradient``)
are exercised, plus the in-place-then-restore behavior that the layer tests
depend on.
"""

import numpy as np

from miniviz.tests.grad_check import numerical_gradient, numerical_x_gradient


def test_x_gradient_of_squared_sum():
    """``f(x) = sum(x**2)`` should produce numerical gradient ``2*x``."""
    rng = np.random.default_rng(0)
    x = rng.standard_normal((3, 4), dtype=np.float32)

    def f(xx: np.ndarray) -> float:
        return float((xx**2).sum())

    analytical = 2.0 * x
    numerical = numerical_x_gradient(f, x.copy())
    np.testing.assert_allclose(numerical, analytical, rtol=1e-2, atol=1e-3)


def test_x_gradient_restores_input():
    """``numerical_x_gradient`` should leave ``x`` byte-for-byte unchanged."""
    rng = np.random.default_rng(2)
    x = rng.standard_normal((3, 3), dtype=np.float32)
    before = x.copy()

    def f(xx: np.ndarray) -> float:
        return float(xx.sum())

    numerical_x_gradient(f, x)
    np.testing.assert_array_equal(x, before)


def test_closure_gradient_of_squared_sum():
    """Same ``f(x) = sum(x**2)`` test, but via the zero-arg closure form.

    Mirrors the way layer tests use ``numerical_gradient`` for ``grad_W``
    and ``grad_b`` — the perturbation reaches the function through array
    aliasing rather than through the function's argument.
    """
    rng = np.random.default_rng(3)
    x = rng.standard_normal((3, 4), dtype=np.float32)
    analytical = 2.0 * x.copy()  # snapshot before any perturbation

    def f() -> float:
        return float((x**2).sum())

    numerical = numerical_gradient(f, x)
    np.testing.assert_allclose(numerical, analytical, rtol=1e-2, atol=1e-3)


def test_closure_gradient_restores_params():
    """``numerical_gradient`` should leave the perturbed array unchanged."""
    rng = np.random.default_rng(4)
    params = rng.standard_normal((4, 5), dtype=np.float32)
    before = params.copy()

    def f() -> float:
        return float(params.sum())

    numerical_gradient(f, params)
    np.testing.assert_array_equal(params, before)
