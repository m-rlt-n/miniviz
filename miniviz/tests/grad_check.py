"""Numerical gradient checker.

Provides finite-difference estimates of ``df/dx`` using central differences:

    f'(x_i) ≈ [f(x_i + eps) - f(x_i - eps)] / (2*eps)
"""

from collections.abc import Callable

import numpy as np


def numerical_x_gradient(
    f: Callable[[np.ndarray], float],
    x: np.ndarray,
    eps: float = 1e-3,
) -> np.ndarray:
    """Estimate ``df/dx`` at ``x`` using central differences.

    Perturbs each element of ``x`` by ``±eps`` in turn, passes the perturbed
    array to ``f``, and forms ``(f(x+eps) - f(x-eps)) / (2*eps)``. Use this
    when ``f`` accepts ``x`` directly as an argument (e.g. when testing the
    gradient of a layer with respect to its input).

    ``x`` is mutated in place during evaluation and restored before return,
    so the caller's array is left unchanged afterward.

    Args:
        f: A function mapping an ndarray of shape ``x.shape`` to a scalar
            loss. Receives the perturbed array on each call.
        x: Point at which to evaluate the gradient. Transiently perturbed
            in place.
        eps: Perturbation magnitude. Default ``1e-3`` is sized for float32
            inputs; for float64, ``eps=1e-5`` is tighter.

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


def numerical_gradient(
    f: Callable[[], float],
    params: np.ndarray,
    eps: float = 1e-3,
) -> np.ndarray:
    """Estimate ``df/dparams`` at ``params`` using central differences.

    Perturbs each element of ``params`` by ``±eps`` in turn, calls the
    zero-arg ``f``, and forms ``(f_plus - f_minus) / (2*eps)``. Use this
    when ``f`` doesn't take ``params`` as an argument but reads it
    indirectly — typically when ``params`` is something like
    ``layer.weights.data`` and ``f`` is a closure that runs
    ``layer.forward(fixed_input).sum()``.

    The aliasing is what makes this work: ``params`` and whatever object
    ``f`` reads from must be the *same* numpy buffer, so that mutating
    ``params`` here is visible through the closure inside ``f``.

    ``params`` is mutated in place during evaluation and restored before
    return.

    Args:
        f: A zero-argument function that returns a scalar loss. Closes over
            whatever state it needs (typically the layer and a fixed input).
        params: Array to differentiate with respect to. Transiently
            perturbed in place; must share its buffer with whatever ``f``
            reads.
        eps: Perturbation magnitude. Default ``1e-3`` is sized for float32
            inputs; for float64, ``eps=1e-5`` is tighter.

    Returns:
        Array of the same shape and dtype as ``params`` containing the
        central-difference estimate of ``df/dparams`` at each element.
    """
    grad = np.zeros_like(params)
    for idx in np.ndindex(params.shape):
        original = params[idx]
        params[idx] = original + eps
        loss_plus = float(f())
        params[idx] = original - eps
        loss_minus = float(f())
        params[idx] = original
        grad[idx] = (loss_plus - loss_minus) / (2.0 * eps)

    return grad
