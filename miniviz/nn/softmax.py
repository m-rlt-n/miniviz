"""Softmax layer for nn."""

import attr
import numpy as np

from miniviz.nn.module import Module

@attr.define(kw_only=True)
class SoftMax(Module):
    """Differentiable softmax layer.

    ``y = [e^(z_i)]/[summation from j=1 to M of [e^(z_j)]``

    _p: Private cache of the last set of probabilities output by 
        ``forward``. ``None`` until ``forward`` has run.    
    """
    _p: np.ndarray | None = attr.field(init=False, default=None)

    def forward(self, x: np.ndarray) -> np.ndarray:
        x_shifted = x - x.max(axis=-1, keepdims=True)
        e = np.exp(x_shifted)
        p = e / e.sum(axis=-1, keepdims=True)
        self._p = p
        return p
    
    def backward(self, grad_y: np.ndarray) -> np.ndarray:
        ...
        