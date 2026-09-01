# Copyright (c) AIoWay Authors - All Rights Reserved

"The module for NMF."

from collections import abc as cabc

import torch
from torch import nn, optim

__all__ = ["NmfTrainer"]


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
