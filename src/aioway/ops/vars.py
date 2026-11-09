# Copyright (c) AIoWay Authors - All Rights Reserved

import typing

import torch

from .ops import Op, expr_dcls

__all__ = ["TensorOp"]


@expr_dcls
class TensorOp(Op):
    """
    The `Op` that just holds a `torch.Tensor`.
    """

    tensor: torch.Tensor
    """
    The tensor that this `Op` holds.
    """

    @typing.override
    def execute(self) -> torch.Tensor:
        return self.tensor

    @typing.override
    def deps(self) -> tuple[()]:
        return ()
