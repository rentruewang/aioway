# Copyright (c) AIoWay Authors - All Rights Reserved

import abc
import functools
import operator
import typing
from collections import abc as cabc

import torch
from torch import ops

from aioway._torch import is_float_tensor
from aioway._types import dcls_no_repr

from .fate import Fate

__all__ = [
    "AddTensor",
    "AddScalar",
    "SubTensor",
    "SubScalar",
    "MulTensor",
    "MulScalar",
    "DivTensor",
    "DivScalar",
    "EqTensor",
    "EqScalar",
    "NeTensor",
    "NeScalar",
    "GeTensor",
    "GeScalar",
    "GtTensor",
    "GtScalar",
    "LeTensor",
    "LeScalar",
    "LtTensor",
    "LtScalar",
]

Scalar = int | float | bool


@dcls_no_repr
class _BinaryTensorUFunc(Fate, abc.ABC):
    BINARY: typing.ClassVar[cabc.Callable[..., typing.Any]]

    self: torch.Tensor
    other: torch.Tensor
    alpha: float = 1

    def __post_init__(self):
        if not isinstance(self.self, torch.Tensor):
            raise TypeError(type(self.self))

        if not isinstance(self.other, torch.Tensor):
            raise TypeError(type(self.other))

    @typing.override
    def ok(self) -> bool:
        try:
            _ = torch.broadcast_shapes(self.self.shape, self.other.shape)
            return True
        except RuntimeError:
            return False

    @typing.override
    def do(self) -> torch.Tensor:
        return self.BINARY(self.self, self.other * self.alpha)

    @typing.final
    @typing.override
    def cost(self) -> int:
        return self._shape.numel()

    @functools.cached_property
    def _shape(self) -> torch.Size:
        return torch.broadcast_shapes(self.self.shape, self.other.shape)


@dcls_no_repr
class _BinaryScalarUFunc(Fate, abc.ABC):
    BINARY: typing.ClassVar[cabc.Callable[..., typing.Any]]

    self: torch.Tensor
    other: Scalar

    def __post_init__(self):
        if not isinstance(self.self, torch.Tensor):
            raise TypeError(type(self.self))

    @typing.override
    def ok(self) -> bool:
        return True

    @typing.override
    def do(self) -> torch.Tensor:
        return self.BINARY(self.self, self.other)

    @typing.final
    @typing.override
    def cost(self) -> int:
        return self.self.numel()


class AddTensor(_BinaryTensorUFunc):
    KEY = ops.aten.add.Tensor
    BINARY = operator.add


class AddScalar(_BinaryScalarUFunc):
    KEY = ops.aten.add.Scalar
    BINARY = operator.add


class SubTensor(_BinaryTensorUFunc):
    KEY = ops.aten.sub.Tensor
    BINARY = operator.sub


class SubScalar(_BinaryScalarUFunc):
    KEY = ops.aten.sub.Scalar
    BINARY = operator.sub


class MulTensor(_BinaryTensorUFunc):
    KEY = ops.aten.mul.Tensor
    BINARY = operator.mul


class MulScalar(_BinaryScalarUFunc):
    KEY = ops.aten.mul.Scalar
    BINARY = operator.mul


def _tensor_div(self: torch.Tensor, other: torch.Tensor):
    if is_float_tensor(other):
        return self / other

    else:
        return self // other


class DivTensor(_BinaryTensorUFunc):
    IR = ops.aten.div.Tensor
    BINARY = _tensor_div


def _scalar_div(self: torch.Tensor, other: Scalar, /):
    if isinstance(other, float):
        return self / other

    else:
        return self // other


class DivScalar(_BinaryScalarUFunc):
    IR = ops.aten.div.Scalar
    BINARY = _scalar_div


class EqTensor(_BinaryTensorUFunc):
    IR = ops.aten.eq.Tensor
    BINARY = operator.eq


class EqScalar(_BinaryScalarUFunc):
    IR = ops.aten.eq.Scalar
    BINARY = operator.eq


class NeTensor(_BinaryTensorUFunc):
    IR = ops.aten.ne.Tensor
    BINARY = operator.ne


class NeScalar(_BinaryScalarUFunc):
    IR = ops.aten.ne.Scalar
    BINARY = operator.ne


class GeTensor(_BinaryTensorUFunc):
    IR = ops.aten.ge.Tensor
    BINARY = operator.ge


class GeScalar(_BinaryScalarUFunc):
    IR = ops.aten.ge.Scalar
    BINARY = operator.ge


class GtTensor(_BinaryTensorUFunc):
    IR = ops.aten.gt.Tensor
    BINARY = operator.gt


class GtScalar(_BinaryScalarUFunc):
    IR = ops.aten.gt.Scalar
    BINARY = operator.gt


class LeTensor(_BinaryTensorUFunc):
    IR = ops.aten.le.Tensor
    BINARY = operator.le


class LeScalar(_BinaryScalarUFunc):
    IR = ops.aten.le.Scalar
    BINARY = operator.le


class LtTensor(_BinaryTensorUFunc):
    IR = ops.aten.lt.Tensor
    BINARY = operator.lt


class LtScalar(_BinaryScalarUFunc):
    IR = ops.aten.lt.Scalar
    BINARY = operator.lt
