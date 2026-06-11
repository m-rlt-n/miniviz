"""Softmax layer for nn."""

import attr
import numpy as np

from miniviz.nn.module import Module


@attr.define(kw_only=True)
class Softmax(Module):
    """Differentiable softmax layer.

    ``p_i = exp(z_i - max(z)) / sum_j exp(z_j - max(z))``

    Attrs:
        _p: Private cache of the last set of probabilities output by
            ``forward``. ``None`` until ``forward`` has run.
    """

    _p: np.ndarray | None = attr.field(init=False, default=None)

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Forward method for softmax.

        1. Shift values in input array to be at most 0 (to protect from
        overflow)
        2. Exponentiate the full aray.
        3. Normalize cell values using the rowwise sum.

        Args:
            x: an array of logits returned by upstream layers.

        Returns:
            The probability of each class, shape ``(N, K)``
        """
        x_shifted = x - x.max(axis=-1, keepdims=True)
        e = np.exp(x_shifted)
        p = e / e.sum(axis=-1, keepdims=True)
        self._p = p
        return p

    def backward(self, grad_y: np.ndarray) -> np.ndarray:
        """Backward method for softmax.

        ``dL/dz = p * (dL/dp - sum(p * dL/dp))`` where the sum is over the
        last axis.

        Args:
            grad_y: Upstream gradient of the loss w.r.t. the softmax
                probabilities, same shape as the array returned by
                ``forward``.

        Returns:
            The gradient of the loss w.r.t. the input, shape ``(N, K)``
        """
        s = (self._p * grad_y).sum(axis=-1, keepdims=True)
        return self._p * (grad_y - s)
