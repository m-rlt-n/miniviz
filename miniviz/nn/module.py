"""Abstract base class for differentiable layers and networks."""

from abc import ABC, abstractmethod
from collections.abc import Iterator

import numpy as np

from miniviz.nn.parameter import Parameter


class Module(ABC):
    """Abstract base for any differentiable building block."""

    @abstractmethod
    def forward(self, x: np.ndarray) -> np.ndarray:
        """Run the forward pass. Takes the input array and returns the output array. It
        caches whatever the backward pass will need (typically the input ``x``
        and any intermediate activations) as instance attributes.

        Args:
            x: Input array.

        Returns:
            Output array.
        """

    @abstractmethod
    def backward(self, grad_y: np.ndarray) -> np.ndarray:
        """Run the backward pass. Takes the gradient of the loss with respect to this
        module's *output* and returns the gradient with respect to its *input*.

        Accumulates parameter gradients as a side effect (``param.grad +=
        ...``) and returns the gradient with respect to the input.

        Args:
            grad_y: Gradient of the loss with respect to this module's output.

        Returns:
            Gradient of the loss with respect to this module's input, shaped
            like the ``x`` that was passed to `forward`.
        """

    def parameters(self) -> Iterator[Parameter]:
        """Yield the learnable parameters of this module.

        Default: no parameters (suitable for stateless layers like ``ReLU``).

        Yields:
            Each `Parameter` instance owned by this module.
        """
        return iter(())

    def zero_grad(self) -> None:
        """Reset every parameter's gradient buffer to zero, in place.

        Resetting accumulated grads between training steps is the optimizer's
        responsibility; `zero_grad` is a convenience that does it for you.
        """
        for p in self.parameters():
            p.grad[...] = 0.0

    def __call__(self, x: np.ndarray) -> np.ndarray:
        """Invoke `forward` via call syntax (``layer(x)``)."""
        return self.forward(x)
