"""Softmax layer for nn."""

import attr
import numpy as np

from miniviz.nn.module import Module

@attr.define(kw_only=True)
class SoftMax(Module):
    """Differentiable softmax layer.

    ``y = [e^(z_i)]/[summation from j=1 to K of [e^(z_j)]``

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
        
        Returns: 
            The probability of each class, shape ``(N, K)``
        """
        x_shifted = x - x.max(axis=-1, keepdims=True)
        e = np.exp(x_shifted)
        p = e / e.sum(axis=-1, keepdims=True)
        self._p = p
        return p
    
    def backward(self, grad_p: np.ndarray) -> np.ndarray:
        """Backward method for softmax.

        Returns: 
            The gradient of the loss w.r.t. the input, shape ``(N, K)``
        """
        s = (self._p * grad_p).sum(axis=-1, keepdims=True)
        return self._p * (grad_p - s)        
        