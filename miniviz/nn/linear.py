"""Linear layer for nn"""

import attr
import numpy as np
from collections.abc import Iterator

from miniviz.nn.module import Module
from miniviz.nn.parameter import Parameter
from miniviz.ops.matmul import matmul

@attr.define(kw_only=True)
class Linear(Module):
    """Differentiable class for linear layer in nn.
    
    Attrs: 
        _x: Private attribute to cache last seen x values (N, K)
        weights: Weights learned for linear layer (K, M)
        bias: Intercept parameter for linear layer (M,)
    """
    input_size: int
    output_size: int
    weights: Parameter = attr.field(init=False)
    bias: Parameter = attr.field(init=False)
    _x: np.ndarray | None = attr.field(init=False, default=None)

    def __attrs_post_init__(self):
        """Initilize weights and bias
        """
        scale = np.sqrt(2.0 / self.input_size).astype(np.float32)
        W = (
                np.random.randn(
                    self.input_size, 
                    self.output_size,
                ).astype(np.float32) * scale
            )
        b = np.zeros(
                self.output_size, 
                dtype=np.float32,
            )
        self.weights = Parameter(data=W)
        self.bias = Parameter(data=b)

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Forward method for linear layer. 
        y = x @ W + b

        Args: 
            x: The input array (N, K)

        Returns: 
            The output array (N, M)
        """
        self._x = x
        return matmul(self._x, self.weights.data) + self.bias.data

    def backward(self, grad_y: np.ndarray) -> np.ndarray:
        """Backward method for linear layer. 
        Updates values for the gradient w.r.t the wieghts (K, M) + bias. (M,)

        Args:
            grad_y: Upstream gardient of the loss w.r.t. our output (N, M)
        
        Returns: 
            The gradient of w.r.t. the input (N, K)
        """
        self.weights.grad += matmul(self._x.T, grad_y)
        self.bias.grad += grad_y.sum(axis=0)
        return matmul(grad_y, self.weights.data.T)
    
    def parameters(self) -> Iterator[Parameter]:
        return iter([self.weights, self.bias])


        


    
