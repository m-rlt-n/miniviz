"""NumPy wrapper around the naive C matmul kernel.

The wrapper handles:
 - validating shapes, 
 - coercing inputs to C-contiguous float32, 
 - allocating the output, 
 - calling the C function with ctypes pointers.

The C kernel itself is dtype-locked to float32 — that's the precision we use
throughout the library, matching most modern DL practice.
"""

import ctypes

import numpy as np

from miniviz._backend import lib


def _as_f32_contig(a: np.ndarray) -> np.ndarray:
    """Return ``a`` as a C-contiguous float32 ndarray, copying only if needed."""
    return np.ascontiguousarray(a, dtype=np.float32)


def _f32_ptr(a: np.ndarray):
    """Return a ``float *`` pointer to the underlying buffer of an ndarray."""
    return a.ctypes.data_as(ctypes.POINTER(ctypes.c_float))


def matmul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Compute ``a @ b`` for 2D arrays using the naive C kernel.

    Inputs are coerced to float32 if they are not already; the result is also
    float32. Non-contiguous inputs are copied to a contiguous buffer.

    Args:
        a: 2D array of shape ``(M, K)``.
        b: 2D array of shape ``(K, N)``.

    Returns:
        2D float32 array of shape ``(M, N)`` containing ``a @ b``.
    """
    if a.ndim != 2 or b.ndim != 2:
        raise ValueError(
            f"matmul expects 2D arrays, got shapes {a.shape} and {b.shape}"
        )

    M, K = a.shape
    K2, N = b.shape
    if K != K2:
        raise ValueError(f"matmul shape mismatch: ({M}, {K}) @ ({K2}, {N})")

    a_f32 = _as_f32_contig(a)
    b_f32 = _as_f32_contig(b)
    c = np.zeros((M, N), dtype=np.float32)

    lib().matmul_naive_f32(
        _f32_ptr(a_f32),
        _f32_ptr(b_f32),
        _f32_ptr(c),
        ctypes.c_size_t(M),
        ctypes.c_size_t(N),
        ctypes.c_size_t(K),
    )
    return c
