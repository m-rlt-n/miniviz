"""ReLU activation for nn."""

import attr
import numpy as np

from miniviz.nn.module import Module


@attr.define(kw_only=True)
class ReLU(Module):
    """Differentiable rectified linear unit.

    differentiable at ``x = 0``; by convention we treat that point as
    "non-positive" and zero the gradient there, matching every mainstream
    framework.

    Attrs:
        _mask: Private cache of the positivity mask from the last ``forward``
    """

    _mask: np.ndarray | None = attr.field(init=False, default=None)

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Forward pass: elementwise ``max(0, x)``.

        Args:
            x: Input array of any shape.

        Returns:
            Output array, same shape and dtype as ``x``.
        """
        self._mask = x > 0
        return np.maximum(x, 0)

    def backward(self, grad_y: np.ndarray) -> np.ndarray:
        """Backward pass: zero out gradient where the input was non-positive.

        ``grad_x = grad_y * mask`` where ``mask`` was cached during fwd.

        Args:
            grad_y: Upstream gradient of the loss w.r.t. our output. Same
                shape as the ``x`` that was passed to ``forward``.

        Returns:
            Gradient of the loss w.r.t. the input, same shape as the input.
        """
        return grad_y * self._mask
