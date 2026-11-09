# Copyright (c) AIoWay Authors - All Rights Reserved

import typing

import torch

from .ops import Expr, expr_dcls

__all__ = ["TensorExpr"]


@expr_dcls
class TensorExpr(Expr):
    """
    The `Expr` that just holds a `torch.Tensor`.
    """

    tensor: torch.Tensor
    """
    The tensor that this `Expr` holds.
    """

    @typing.override
    def execute(self) -> torch.Tensor:
        return self.tensor

    @typing.override
    def deps(self) -> tuple[()]:
        return ()
