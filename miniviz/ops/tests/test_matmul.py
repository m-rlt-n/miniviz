"""Correctness tests for the naive C matmul.

These compare miniviz.matmul against numpy.matmul on a variety of shapes
and dtypes. The whole point of phase one is to prove that the C↔Python pipe
works end to end — if these pass, every kernel we add later just needs to
follow the same pattern.
"""

import numpy as np
import pytest

from miniviz.ops.matmul import matmul


def _allclose_f32(actual, expected):
    """Tolerance appropriate for float32 dot products of moderate size."""
    np.testing.assert_allclose(actual, expected, rtol=1e-3, atol=1e-4)


def test_matmul_small_rectangular():
    rng = np.random.default_rng(0)
    a = rng.standard_normal((3, 4)).astype(np.float32)
    b = rng.standard_normal((4, 5)).astype(np.float32)
    _allclose_f32(matmul(a, b), a @ b)


def test_matmul_square():
    rng = np.random.default_rng(1)
    a = rng.standard_normal((64, 64)).astype(np.float32)
    b = rng.standard_normal((64, 64)).astype(np.float32)
    _allclose_f32(matmul(a, b), a @ b)


def test_matmul_irregular_shapes():
    rng = np.random.default_rng(2)
    a = rng.standard_normal((17, 31)).astype(np.float32)
    b = rng.standard_normal((31, 23)).astype(np.float32)
    _allclose_f32(matmul(a, b), a @ b)


def test_matmul_thin_and_wide():
    """Edge cases: 1xK @ Kx1 (dot product) and Mx1 @ 1xN (outer product)."""
    rng = np.random.default_rng(3)
    a = rng.standard_normal((1, 50)).astype(np.float32)
    b = rng.standard_normal((50, 1)).astype(np.float32)
    _allclose_f32(matmul(a, b), a @ b)

    c = rng.standard_normal((10, 1)).astype(np.float32)
    d = rng.standard_normal((1, 7)).astype(np.float32)
    _allclose_f32(matmul(c, d), c @ d)


def test_matmul_identity():
    rng = np.random.default_rng(4)
    a = rng.standard_normal((8, 8)).astype(np.float32)
    eye = np.eye(8, dtype=np.float32)
    _allclose_f32(matmul(a, eye), a)
    _allclose_f32(matmul(eye, a), a)


def test_matmul_zero_input():
    a = np.zeros((5, 5), dtype=np.float32)
    b = np.ones((5, 5), dtype=np.float32)
    np.testing.assert_array_equal(matmul(a, b), np.zeros((5, 5), dtype=np.float32))


def test_matmul_dtype_coerced_from_float64():
    """float64 inputs should be silently coerced to float32."""
    a = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64)
    b = np.array([[5.0, 6.0], [7.0, 8.0]], dtype=np.float64)
    expected = (a @ b).astype(np.float32)
    _allclose_f32(matmul(a, b), expected)


def test_matmul_non_contiguous_input():
    """Non-contiguous views should still work (we copy them internally)."""
    rng = np.random.default_rng(5)
    big = rng.standard_normal((20, 20)).astype(np.float32)
    a = big[::2, ::2]  # non-contiguous slice
    b = big[1::2, 1::2]
    _allclose_f32(matmul(a, b), a @ b)


def test_matmul_shape_mismatch_raises():
    a = np.zeros((3, 4), dtype=np.float32)
    b = np.zeros((5, 6), dtype=np.float32)
    with pytest.raises(ValueError, match="shape mismatch"):
        matmul(a, b)


def test_matmul_wrong_ndim_raises():
    a = np.zeros((3,), dtype=np.float32)
    b = np.zeros((3, 4), dtype=np.float32)
    with pytest.raises(ValueError, match="2D arrays"):
        matmul(a, b)
