"""Correctness tests for the ReLU activation.

Forward: positive values pass through, non-positive values become zero.
Backward: gradient passes through where input was positive, zero elsewhere,
verified both with hand-rolled cases and against a numerical gradient
estimate.

A note on the gradient check: ReLU is non-differentiable at ``x = 0`` (a
"kink" in the function). The numerical gradient breaks down for any input
within ``eps`` of the kink, because the central difference straddles the
kink and averages the two one-sided derivatives. To keep the check honest,
we deliberately construct inputs that are bounded well away from zero.
"""

import numpy as np

from miniviz.nn.relu import ReLU
from miniviz.tests.grad_check import numerical_x_gradient


def test_forward_zeros_non_positive():
    """Negative and zero inputs map to zero; positive inputs pass through."""
    layer = ReLU()
    x = np.ones((3,3))
    x -= np.eye(3)
    y = layer(x)
    expected = np.array([
        [0, 1, 1], [1, 0, 1], [1, 1, 0]
    ], dtype=np.float32)
    np.testing.assert_array_equal(y, expected)


def test_forward_shape_preserved():
    """Output shape equals input shape for any rank."""
    layer = ReLU()
    rng = np.random.default_rng(0)
    for shape in [(5,), (3, 4), (2, 3, 4), (2, 3, 4, 5)]:
        x = rng.standard_normal(shape, dtype=np.float32)
        assert layer(x).shape == shape


def test_forward_dtype_preserved():
    """Output dtype matches input dtype (float32)."""
    layer = ReLU()
    x = np.array([1.0, -1.0], dtype=np.float32)
    assert layer(x).dtype == np.float32


def test_backward_passes_through_positive():
    """Where the input was positive, the upstream gradient is unchanged."""
    layer = ReLU()
    x = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    layer(x)
    grad_y = np.array([0.5, 0.7, 0.9], dtype=np.float32)
    np.testing.assert_array_equal(layer.backward(grad_y), grad_y)


def test_backward_zeroes_non_positive():
    """Where the input was non-positive, the gradient is zero."""
    layer = ReLU()
    x = np.array([-1.0, 0.0, 1.0], dtype=np.float32)
    layer(x)
    grad_y = np.array([10.0, 20.0, 30.0], dtype=np.float32)
    expected = np.array([0.0, 0.0, 30.0], dtype=np.float32)
    np.testing.assert_array_equal(layer.backward(grad_y), expected)


def test_backward_matches_numerical():
    """Analytical ``grad_x`` matches a finite-difference estimate.

    Inputs are constructed so that ``|x| >= 0.5`` everywhere, far enough
    from the kink at ``x = 0`` that no ``±eps`` perturbation crosses it.
    """
    layer = ReLU()
    rng = np.random.default_rng(42)
    signs = rng.choice([-1.0, 1.0], size=(3, 4)).astype(np.float32)
    magnitudes = rng.uniform(0.5, 2.0, size=(3, 4)).astype(np.float32)
    x = signs * magnitudes
    grad_y = rng.standard_normal((3, 4), dtype=np.float32)

    layer.forward(x)
    analytical = layer.backward(grad_y)

    def loss(xx: np.ndarray) -> float:
        return float((layer.forward(xx) * grad_y).sum())

    numerical = numerical_x_gradient(loss, x.copy())
    np.testing.assert_allclose(analytical, numerical, rtol=1e-2, atol=1e-3)


def test_no_parameters():
    """ReLU has no learnable parameters."""
    layer = ReLU()
    assert list(layer.parameters()) == []
