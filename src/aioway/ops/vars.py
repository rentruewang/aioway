# Copyright (c) AIoWay Authors - All Rights Reserved

import typing

import torch

from aioway.t import AnyDomain, Domain, TensorDomain

from .ops import Op

__all__ = ["ValueOp", "TensorOp"]


class ValueOp[T](Op[T]):
    """
    The `Op` that wraps any value.
    """

    value: T
    "Any value that is wrapped."

    @property
    @typing.override
    def __domain_deps__(self) -> list[Domain]:
        return []

    @typing.override
    def _infer_domain(self) -> AnyDomain:
        return AnyDomain()

    @typing.override
    def execute(self) -> T:
        return self.value

    @property
    @typing.override
    def __deps__(self) -> tuple[()]:
        return ()


class TensorOp(Op[torch.Tensor]):
    """
    The `Op` that just holds a `torch.Tensor`.
    """

    tensor: torch.Tensor
    """
    The tensor that this `Op` holds.
    """

    @property
    @typing.override
    def __domain_deps__(self) -> list[Domain]:
        return []

    @typing.override
    def _infer_domain(self) -> TensorDomain:
        return TensorDomain()

    @typing.override
    def execute(self) -> torch.Tensor:
        return self.tensor

    @property
    @typing.override
    def __deps__(self) -> tuple[()]:
        return ()
