# Copyright (c) AIoWay Authors - All Rights Reserved

"The module for NMF."

import typing
from collections import abc as cabc

import tensordict as td
import torch
from torch import nn, optim

from aioway.emits import FuncEmitter
from aioway.errors import re_raise_func
from aioway.spaces import (
    Attr,
    AttrDict,
    BoxSpace,
    Space,
    TdictSpace,
    TensorSpace,
    space_dcls,
)

__all__ = ["NMFSpace", "PairSpace", "NMFTrainerModule", "train_nmf"]


@typing.final
@space_dcls
class NMFSpace(TensorSpace):
    """
    NMF is 1 space duplicated, or 2 spaces of the same type.
    """

    hidden: int
    "The hidden latent size."

    num_rows: int
    "The number of rows to constrain."

    num_cols: int
    "The number of cols to constrain."

    @typing.override
    @re_raise_func(AssertionError, ValueError)
    def _check_attr(self, attr: Attr) -> None:
        # Batch, m, n
        assert attr.ndim == 3

        assert attr.shape[1] == self.num_rows
        assert attr.shape[2] == self.num_cols

    @re_raise_func(AssertionError, ValueError)
    def _check_data(self, data: torch.Tensor) -> None:
        assert torch.all(data >= 0).item()

    @typing.override
    def _sample_n(self, n: int) -> torch.Tensor:
        rows = self.num_rows or 3
        cols = self.num_rows or 5
        sample = torch.rand(n, rows, cols)
        return sample


class PairSpace(TdictSpace):
    """
    A pair of space (later, should adapt to use the one loss function uses).
    """

    input: TensorSpace
    "Input space of the pair."

    target: TensorSpace
    "Target space of the pair."

    def _check_attrs(self, attrs: AttrDict) -> None:
        assert len(attrs) == 2
        assert attrs.keys() == {"input", "target"}

    def _check_data(self, data: td.TensorDict) -> None:
        pass


class NMFTrainerModule(nn.Module):
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


@FuncEmitter
def train_nmf(observation_space: Space, action_space: Space) -> nn.Module:
    if not isinstance(observation_space, NMFSpace):
        return NotImplemented

    if not (isinstance(action_space, BoxSpace) and action_space.ndim == 0):
        return NotImplemented

    hidden = observation_space.hidden
    rows = observation_space.num_rows
    cols = observation_space.num_cols

    left = nn.Parameter(torch.empty(hidden, rows))
    right = nn.Parameter(torch.empty(hidden, cols))
    loss = nn.MSELoss()
    optim_type = optim.Adam

    return NMFTrainerModule(left, right, loss, optim_type)
