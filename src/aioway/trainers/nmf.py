# Copyright (c) AIoWay Authors - All Rights Reserved

"The module for NMF."

from collections import abc as cabc

import torch
from torch import nn, optim
from torchrl.data import tensor_specs as tspecs

from aioway.nets import Emitter, emitter_dcls
from aioway.tspecs import Space, TSpecSpace

__all__ = ["NmfTrainer", "NmfEmitter"]


class NmfTrainer(nn.Module):
    def __init__(
        self,
        left: nn.Parameter,
        right: nn.Parameter,
        loss: nn.Module,
        optim_type: cabc.Callable[..., optim.Optimizer],
    ) -> None:
        """
        Args:
            h: The hidden size.
            m: The number of features for the LHS.
            n: The number of features for the RHS.
        """

        super().__init__()

        self.left = left
        self.right = right
        self.loss = loss
        self.optim = optim_type([self.left, self.right])

        h_left, m = self.left.shape
        h_right, n = self.right.shape
        assert h_left == h_right, {"left": h_left, "right": h_right}

    def forward(self, matrix: torch.Tensor) -> None:
        assert matrix.ndim == 3
        assert matrix.shape[1] == self._m
        assert matrix.shape[2] == self._n

        pred = torch.einsum("hm,hn->mn", self.left, self.right)
        loss = self.loss(pred, matrix)

        self.optim.zero_grad()
        loss.backward()
        self.optim.step()


@emitter_dcls
class NmfEmitter(Emitter):

    hidden: int
    "The hidden latent size."

    def __call__(self, observ: Space, action: Space) -> nn.Module:
        if not isinstance(observ, TSpecSpace):
            return NotImplemented
        if not isinstance(action, TSpecSpace):
            return NotImplemented

        if not (isinstance(observ.spec, tspecs.Unbounded) and observ.ndim == 2):
            return NotImplemented

        if not (
            isinstance(action.spec, tspecs.UnboundedContinuous) and action.ndim == 0
        ):
            return NotImplemented

        hidden = self.hidden
        rows, cols = observ.shape

        left = nn.Parameter(torch.empty(hidden, rows))
        right = nn.Parameter(torch.empty(hidden, cols))
        loss = nn.MSELoss()
        optim_type = optim.Adam

        return NmfTrainer(left, right, loss, optim_type)
