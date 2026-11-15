# Copyright (c) AIoWay Authors - All Rights Reserved

import typing

import torch
from torch import nn

from .ops import Op
from .thunks import ModuleOp

__all__ = ["LayerOp"]


class LayerOp(ModuleOp):
    """
    A `torch.nn` layer applied to one input: `(input: tensor) -> tensor`.

    Losses are not layers, since they take an input and a target.
    """

    module: nn.Module
    "The layer."

    input: Op[torch.Tensor]
    "The `Op` producing the layer's input."

    @typing.override
    def execute(self) -> torch.Tensor:
        return self.module(self.input())

    @property
    @typing.override
    def nn(self) -> nn.Module:
        return self.module
