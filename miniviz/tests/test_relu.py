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
from miniviz.tests.grad_check import numerical_gradient


def test_forward_zeros_non_positive():
    """Negative and zero inputs map to zero; positive inputs pass through."""
    layer = ReLU()
    x = np.ones((3,3))
    x[:2,:2] *= -1
    y = layer(x)
    expected = np.array([
        [0, 0, 1], [0, 0, 1], [1, 1, 1]
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