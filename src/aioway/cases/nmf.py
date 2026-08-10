# Copyright (c) AIoWay Authors - All Rights Reserved

"The module for NMF."

from collections import abc as cabc

import torch
from torch import nn, optim
from torchrl import data as rldata

from aioway.emits import Emitter, emitter_dcls

__all__ = ["NmfTrainerModule", "NmfEmitter"]


class NmfTrainerModule(nn.Module):
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

    num_rows: int
    "The number of rows to constrain."

    num_cols: int
    "The number of cols to constrain."

    def __call__(
        self, observ: rldata.TensorSpec, action: rldata.TensorSpec
    ) -> nn.Module:
        if not isinstance(observ, rldata.Bounded):
            return NotImplemented

        if not (isinstance(action, rldata.UnboundedContinuous) and action.ndim == 0):
            return NotImplemented

        hidden = self.hidden
        rows = self.num_rows
        cols = self.num_cols

        left = nn.Parameter(torch.empty(hidden, rows))
        right = nn.Parameter(torch.empty(hidden, cols))
        loss = nn.MSELoss()
        optim_type = optim.Adam

        return NmfTrainerModule(left, right, loss, optim_type)
