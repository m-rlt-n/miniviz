"""Correctness tests for the Linear layer.

Forward: output matches a hand-rolled numpy reference at float32 tolerance.

Backward: analytical gradients (grad_x, grad_W, grad_b) match a numerical
gradient computed by central differences. The numerical check is the
authoritative oracle here — if it agrees with what `backward` returns, the
math in the layer is right. The same pattern (and this same helper) will
test every layer we add after this one.
"""

import numpy as np

from miniviz.nn.linear import Linear
from miniviz.tests.grad_check import numerical_gradient


def _build_layer_and_x(
    input_size: int = 4,
    output_size: int = 3,
    batch: int = 2,
    seed: int = 42,
) -> tuple[Linear, np.ndarray]:
    """Construct a deterministic Linear and a deterministic float32 input."""
    layer = Linear(input_size=input_size, output_size=output_size, seed=seed)
    rng = np.random.default_rng(seed + 1)
    x = rng.standard_normal((batch, input_size), dtype=np.float32)
    return layer, x


def test_forward_shape():
    """Forward output should have shape ``(batch, output_size)``."""
    layer, x = _build_layer_and_x(input_size=5, output_size=7, batch=3)
    y = layer(x)
    assert y.shape == (3, 7)


def test_forward_matches_numpy_reference():
    """Forward should compute ``x @ W + b`` within float32 tolerance."""
    layer, x = _build_layer_and_x()
    y = layer(x)
    expected = x @ layer.weights.data + layer.bias.data
    np.testing.assert_allclose(y, expected, rtol=1e-3, atol=1e-4)


def test_backward_grad_x_matches_numerical():
    """``grad_x`` returned by ``backward`` should match a finite-difference estimate."""
    layer, x = _build_layer_and_x()
    y = layer(x)
    grad_y = np.ones_like(y)

    analytical = layer.backward(grad_y)

    # Loss closure: sum over all output elements (so its gradient w.r.t.
    # the output is the all-ones tensor we used for the analytical pass).
    def loss(xx: np.ndarray) -> float:
        return float(layer.forward(xx).sum())

    numerical = numerical_gradient(loss, x.copy())
    np.testing.assert_allclose(analytical, numerical, rtol=1e-2, atol=1e-3)


def test_backward_grad_W_matches_numerical():
    """``weights.grad`` accumulated by ``backward`` should match a numerical estimate."""
    layer, x = _build_layer_and_x()
    y = layer(x)

    layer.zero_grad()
    layer.forward(x)
    layer.backward(np.ones_like(y))
    analytical = layer.weights.grad.copy()

    # Loss as a function of W: x and b are held fixed; numerical_gradient
    # perturbs layer.weights.data in place, and forward reads from it.
    def loss(_unused_W: np.ndarray) -> float:
        return float(layer.forward(x).sum())

    numerical = numerical_gradient(loss, layer.weights.data)
    np.testing.assert_allclose(analytical, numerical, rtol=1e-2, atol=1e-3)


def test_backward_grad_b_matches_numerical():
    """``bias.grad`` accumulated by ``backward`` should match a numerical estimate."""
    layer, x = _build_layer_and_x()
    y = layer(x)

    layer.zero_grad()
    layer.forward(x)
    layer.backward(np.ones_like(y))
    analytical = layer.bias.grad.copy()

    def loss(_unused_b: np.ndarray) -> float:
        return float(layer.forward(x).sum())

    numerical = numerical_gradient(loss, layer.bias.data)
    np.testing.assert_allclose(analytical, numerical, rtol=1e-2, atol=1e-3)


def test_zero_grad_resets_to_zero():
    """``zero_grad`` should reset both ``weights.grad`` and ``bias.grad`` in place."""
    layer, x = _build_layer_and_x()
    y = layer(x)
    layer.backward(np.ones_like(y))

    # Sanity: at least the weights grad accumulated something
    assert layer.weights.grad.any()

    layer.zero_grad()
    assert not layer.weights.grad.any()
    assert not layer.bias.grad.any()


def test_parameters_order():
    """``parameters()`` should yield weights, then bias, in that order."""
    layer, _ = _build_layer_and_x()
    params = list(layer.parameters())
    assert params == [layer.weights, layer.bias]
