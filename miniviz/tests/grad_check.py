"""Numerical gradient checker.

Finite-difference estimate of ``df/dx``, used to verify that a layer's
hand-written ``backward`` matches the truth implied by its ``forward``.
The standard pattern: wrap the layer in a scalar-valued loss closure,
compute the analytical gradient via ``backward``, compute the numerical
gradient via :func:`numerical_gradient`, and assert they're close to
``float32`` tolerance.

Why central differences. The forward-difference estimate ``(f(x+eps) -
f(x)) / eps`` has truncation error proportional to ``eps``; the central
form ``(f(x+eps) - f(x-eps)) / (2*eps)`` has error proportional to
``eps**2``, which is dramatically more accurate for the same step size.
For float32 inputs, the practical sweet spot is ``eps ~ 1e-3`` — large
enough that the subtraction doesn't lose meaningful bits to roundoff,
small enough that truncation error stays at the ``1e-6`` level.
"""

from collections.abc import Callable

import numpy as np


def numerical_gradient(
    f: Callable[[np.ndarray], float],
    x: np.ndarray,
    eps: float = 1e-3,
) -> np.ndarray:
    """Estimate ``df/dx`` at ``x`` using central differences.

    For each element ``x[i, j, ...]`` of ``x``, this perturbs ``x`` by
    ``+eps`` at that position, evaluates ``f``, then by ``-eps``, evaluates
    again, and forms ``(f(x+eps) - f(x-eps)) / (2*eps)``. The result has
    the same shape and dtype as ``x``.

    Implementation note: ``x`` is mutated in place during evaluation and
    restored before each iteration ends and before the function returns.
    ``f`` must therefore not retain references to ``x`` across calls in a
    way that could be confused by these transient perturbations (in
    practice this means: don't cache ``x`` inside ``f`` and read it back
    later).

    Args:
        f: A function mapping an ndarray of shape ``x.shape`` to a scalar
            loss. Must accept the same array object on every call.
        x: Point at which to evaluate the numerical gradient. Will be
            transiently perturbed in place during the call.
        eps: Perturbation magnitude. Default ``1e-3`` is sized for float32
            inputs; for float64 inputs ``eps=1e-5`` is tighter.

    Returns:
        Array of the same shape and dtype as ``x`` containing the
        central-difference estimate of ``df/dx`` at each element.
    """
    grad = np.zeros_like(x)
    for idx in np.ndindex(x.shape):
        original = x[idx]
        x[idx] = original + eps
        loss_plus = float(f(x))
        x[idx] = original - eps
        loss_minus = float(f(x))
        x[idx] = original
        grad[idx] = (loss_plus - loss_minus) / (2.0 * eps)
    return grad
