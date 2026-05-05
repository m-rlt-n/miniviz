"""Learnable parameter container.

A :class:`Parameter` bundles a tensor of values together with a parallel
tensor of accumulated gradients. Layers expose their parameters through
:meth:`miniviz.nn.module.Module.parameters`; the optimizer reads ``data``
and ``grad`` to update the weights.
"""

import attr
import numpy as np


@attr.define(kw_only=True)
class Parameter:
    """A learnable parameter: weight values and their gradient buffer.

    Attributes:
        data: The current parameter values. Read by forward, written by the
            optimizer.
        grad: Accumulated gradient buffer, same shape and dtype as ``data``.
            Defaults to ``np.zeros_like(data)`` if not supplied.
    """

    data: np.ndarray
    grad: np.ndarray = attr.field()

    @grad.default
    def _default_grad(self) -> np.ndarray:
        """Default the gradient buffer to a zeros-like array of ``data``."""
        return np.zeros_like(self.data)
