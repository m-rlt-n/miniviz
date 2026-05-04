"""Loader for the compiled C library.

We use ctypes to ``dlopen`` the shared library and declare the C function
signatures so Python knows how to marshal arguments. Each C kernel we add
later only needs an entry in :func:`_register_signatures` below.

The library lives inside the Python package at ``miniviz/_native/``. That path
is correct in three different scenarios:

1. Local dev — ``make`` drops it at ``miniviz/_native/libminiviz.dylib``;
   editable installs (``uv sync``) import the package from the project root,
   so ``__file__`` resolves there.
2. Wheel install — setuptools' ``package-data`` ships ``_native/*.{so,dylib}``
   inside the wheel; pip installs to ``site-packages/miniviz/_native/``.
3. Source tarball — same package layout once setuptools builds it.

Loading is lazy: ``import miniviz`` won't fail just because the C side hasn't
been built yet. You only see an error when you actually call into a kernel,
and the error message points you at ``make``.
"""

import ctypes
import platform
from pathlib import Path

_NATIVE_DIR = Path(__file__).resolve().parent / "_native"


def _lib_filename() -> str:
    """Return the platform-appropriate shared-library filename."""
    system = platform.system()
    if system == "Darwin":
        return "libminiviz.dylib"
    if system == "Linux":
        return "libminiviz.so"
    raise RuntimeError(f"miniviz: unsupported platform {system!r}")


def _register_signatures(lib: ctypes.CDLL) -> None:
    """Declare ``argtypes``/``restype`` for every C function we expose.

    Args:
        lib: The loaded shared library to attach signatures to.
    """
    # void matmul_naive_f32(
    #     const float *A, const float *B, float *C,
    #     size_t M, size_t N, size_t K
    # );
    lib.matmul_naive_f32.argtypes = [
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_size_t,
        ctypes.c_size_t,
        ctypes.c_size_t,
    ]
    lib.matmul_naive_f32.restype = None


_lib_cache: ctypes.CDLL | None = None


def lib() -> ctypes.CDLL:
    """Return the loaded shared library, loading it on first call.

    Returns:
        The cached :class:`ctypes.CDLL` handle to ``libminiviz``, with all C
        function signatures already registered.

    Raises:
        FileNotFoundError: If the compiled library cannot be located. Run
            ``make`` from the project root to build it.
        RuntimeError: If the current platform is not supported.
    """
    global _lib_cache
    if _lib_cache is not None:
        return _lib_cache

    path = _NATIVE_DIR / _lib_filename()
    if not path.exists():
        raise FileNotFoundError(
            f"miniviz: compiled library not found at {path}. "
            "Run `make` from the project root first."
        )

    loaded = ctypes.CDLL(str(path))
    _register_signatures(loaded)
    _lib_cache = loaded
    return loaded
