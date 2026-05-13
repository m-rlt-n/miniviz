"""ReLU activation for nn."""

import attr
import numpy as np

from miniviz.nn.module import Module


@attr.define(kw_only=True)
class ReLU(Module):
    """Differentiable rectified linear unit.

    Forward: ``y = max(0, x)`` elementwise.

    Backward: the gradient passes through unchanged where the input was
    strictly positive and is zeroed where the input was non-positive —
    formally, ``grad_x = grad_y * (x > 0)``. The function is not
    differentiable at ``x = 0``; by convention we treat that point as
    "non-positive" and zero the gradient there, matching every mainstream
    framework.

    ReLU has no learnable parameters, so it inherits ``Module.parameters``'s
    default empty iterator. The only state it carries between forward and
    backward is a boolean mask of the elements that were positive — cheaper
    than caching the full input, and exactly what the backward pass needs.

    Attrs:
        _mask: Private cache of the positivity mask from the last ``forward``
            call, same shape as the input. ``None`` until ``forward`` has
            run.
    """

    _mask: np.ndarray | None = attr.field(init=False, default=None)

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Forward pass: elementwise ``max(0, x)``.

        Works on inputs of any rank — convolutional feature maps, sequences
        of token embeddings, plain ``(batch, features)`` matrices — because
        the operation is purely elementwise.

        Args:
            x: Input array of any shape.

        Returns:
            Output array, same shape and dtype as ``x``.
        """
        self._mask = x > 0
        return np.maximum(x, 0)

    def backward(self, grad_y: np.ndarray) -> np.ndarray:
        """Backward pass: zero out gradient where the input was non-positive.

        ``grad_x = grad_y * mask`` where ``mask`` was cached during
        ``forward``. Multiplying ``float32 * bool`` yields ``float32`` with
        zeros where the mask is False, ones elsewhere.

        Args:
            grad_y: Upstream gradient of the loss w.r.t. our output. Same
                shape as the ``x`` that was passed to ``forward``.

        Returns:
            Gradient of the loss w.r.t. the input, same shape as the input.
        """
        return grad_y * self._mask
