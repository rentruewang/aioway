# Copyright (c) AIoWay Authors - All Rights Reserved

import typing

import torch

from .ops import Op

__all__ = ["ValueOp", "TensorOp"]


class ValueOp[T](Op[T]):
    """
    The `Op` that wraps any value.
    """

    value: T
    "Any value that is wrapped."

    def execute(self) -> T:
        return self.value

    @typing.override
    def deps(self) -> tuple[()]:
        return ()


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
