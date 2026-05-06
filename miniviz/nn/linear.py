"""Linear layer for nn."""

from collections.abc import Iterator

import attr
import numpy as np

from miniviz.nn.module import Module
from miniviz.nn.parameter import Parameter
from miniviz.ops.matmul import matmul


@attr.define(kw_only=True)
class Linear(Module):
    """Differentiable linear (fully-connected) layer.

    Computes ``y = x @ W + b`` for a batch of inputs. Weights are initialized
    with Kaiming-style scaling (variance ``2 / input_size``).

    Attrs:
        input_size: K, the number of input features.
        output_size: M, the number of output features.
        seed: Optional integer used to seed the weight initialization.
        weights: Learnable weight ``Parameter``, shape ``(K, M)``
        bias: Learnable bias ``Parameter``, shape ``(M,)``
        _x: Private cache of the last input passed to ``forward``, used by
            ``backward``. ``None`` until ``forward`` has run.
    """

    input_size: int
    output_size: int
    seed: int | None = attr.field(default=None)
    weights: Parameter = attr.field(init=False)
    bias: Parameter = attr.field(init=False)
    _x: np.ndarray | None = attr.field(init=False, default=None)

    def __attrs_post_init__(self) -> None:
        """Initialize ``weights`` (Kaiming-scaled normals) and ``bias`` (zeros).
        """
        rng = np.random.default_rng(self.seed)
        scale = np.sqrt(2.0 / self.input_size).astype(np.float32)
        W = rng.standard_normal(
            (self.input_size, self.output_size),
            dtype=np.float32,
        ) * scale
        b = np.zeros(self.output_size, dtype=np.float32)
        self.weights = Parameter(data=W)
        self.bias = Parameter(data=b)

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Forward method for linear layer.

        ``y = x @ W + b``

        Args:
            x: The input array, shape ``(N, K)``.

        Returns:
            The output array, shape ``(N, M)``.
        """
        self._x = x
        return matmul(self._x, self.weights.data) + self.bias.data

    def backward(self, grad_y: np.ndarray) -> np.ndarray:
        """Backward method for linear layer.

        Accumulates parameter gradients into ``self.weights.grad`` and
        ``self.bias.grad`` (using ``+=``), and returns the gradient with
        respect to the input so it can flow upstream.

        Args:
            grad_y: Upstream gradient of the loss w.r.t. our output, shape
                ``(N, M)``.

        Returns:
            The gradient of the loss w.r.t. the input, shape ``(N, K)``.
        """
        self.weights.grad += matmul(self._x.T, grad_y)
        self.bias.grad += grad_y.sum(axis=0)
        return matmul(grad_y, self.weights.data.T)

    def parameters(self) -> Iterator[Parameter]:
        """Yield the layer's learnable parameters (weights, then bias)."""
        return iter([self.weights, self.bias])
