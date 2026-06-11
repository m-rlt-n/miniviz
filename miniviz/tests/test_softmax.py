"""Correctness tests for the Softmax layer."""

import numpy as np

from miniviz.nn.softmax import Softmax
from miniviz.tests.grad_check import numerical_x_gradient


def _naive_softmax(z: np.ndarray) -> np.ndarray:
    """Textbook softmax with no max-subtraction.

    Mathematically equivalent to ``Softmax.forward``, numerically unstable
    for logits with very large magnitudes. Used as a reference oracle on
    well-conditioned inputs only.
    """
    e = np.exp(z)
    return e / e.sum(axis=-1, keepdims=True)


def test_forward_shape_preserved_and_sums_to_one():
    """Output shape matches input shape; last-axis slices sum to 1.

    Covers both the classifier-head shape ``(N, K)`` and the
    attention-shaped ``(B, H, N, N)`` case the layer will see in phase 3.
    """
    layer = Softmax()
    rng = np.random.default_rng(0)
    for shape in [(3, 4), (2, 3, 4, 5)]:
        z = rng.standard_normal(shape, dtype=np.float32)
        p = layer(z)
        assert p.shape == shape
        np.testing.assert_allclose(
            p.sum(axis=-1),
            np.ones(shape[:-1], dtype=np.float32),
            rtol=1e-5,
            atol=1e-6,
        )


def test_forward_dtype_preserved():
    """Output dtype matches input dtype (float32)."""
    layer = Softmax()
    z = np.array([[1.0, 2.0, 3.0]], dtype=np.float32)
    assert layer(z).dtype == np.float32


def test_forward_matches_naive_reference():
    """For well-conditioned logits, forward matches the textbook formula."""
    layer = Softmax()
    rng = np.random.default_rng(2)
    z = rng.standard_normal((3, 5), dtype=np.float32)
    expected = _naive_softmax(z)
    np.testing.assert_allclose(layer(z), expected, rtol=1e-5, atol=1e-6)


def test_forward_stable_for_large_logits():
    """Max-subtraction prevents overflow for logits past float32's exp range.

    ``exp(200)`` is ``inf`` in float32 (the limit is around ``exp(88)``),
    so a naive softmax produces ``nan`` here. With the shift, every
    exponent lands in ``[exp(-1), exp(0)]`` and the result is finite.
    """
    layer = Softmax()
    z = np.array([[200.0, 200.0, 201.0]], dtype=np.float32)
    p = layer(z)
    assert np.all(np.isfinite(p))
    np.testing.assert_allclose(p.sum(axis=-1), 1.0, rtol=1e-5, atol=1e-6)


def test_backward_grad_x_matches_numerical():
    """Analytical ``grad_z`` matches a finite-difference estimate.

    Logits are drawn from ``standard_normal``, so probabilities stay
    bounded away from 0 and 1 — no saturation, no kink, central
    differences are trustworthy.
    """
    layer = Softmax()
    rng = np.random.default_rng(42)
    z = rng.standard_normal((3, 4), dtype=np.float32)
    grad_y = rng.standard_normal((3, 4), dtype=np.float32)

    layer.forward(z)
    analytical = layer.backward(grad_y)

    def loss(zz: np.ndarray) -> float:
        return float((layer.forward(zz) * grad_y).sum())

    numerical = numerical_x_gradient(loss, z.copy())
    np.testing.assert_allclose(analytical, numerical, rtol=1e-2, atol=1e-3)


def test_backward_constant_grad_y_gives_zero():
    """Constant upstream gradient produces zero downstream gradient.

    Softmax is shift-invariant: ``softmax(z + c) == softmax(z)``. The
    dual statement is that if the upstream loss only depends on the
    probabilities through their (constant) sum, then the gradient w.r.t.
    the logits must be zero. With ``grad_p[k] = c`` for all k, the
    backward formula reduces to ``p * (c - c) = 0``.
    """
    layer = Softmax()
    rng = np.random.default_rng(7)
    z = rng.standard_normal((2, 3), dtype=np.float32)
    layer.forward(z)
    grad_y = np.full((2, 3), 0.5, dtype=np.float32)
    grad_z = layer.backward(grad_y)
    np.testing.assert_allclose(grad_z, 0.0, atol=1e-6)


def test_no_parameters():
    """Softmax has no learnable parameters."""
    layer = Softmax()
    assert list(layer.parameters()) == []
