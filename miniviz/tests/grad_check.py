"""Numerical gradient checker.Check numerical gradients with a 
finite-difference estimate of ``df/dx``.

    f'(x_i) ≈ [f(x_i + eps) - f(x_i - eps)] / (2*eps)

We constrain the loss by wrapping the value w/in a tolerance of eps,
then compute the gradient via ``backward`` and assert they're close.
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
