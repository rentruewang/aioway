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
    "TrueDivTensor",
    "TrueDivScalar",
    "FloorDiv",
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
class _BinaryUFunc(Fate, abc.ABC):
    BINARY: typing.ClassVar[cabc.Callable[..., typing.Any]]

    self: torch.Tensor | Scalar
    other: torch.Tensor | Scalar
    alpha: float = 1

    @typing.override
    def ok(self) -> bool:
        return (
            False
            or isinstance(self.self, torch.Tensor)
            or isinstance(self.other, torch.Tensor)
        )

    @typing.override
    def do(self) -> torch.Tensor:
        return self.BINARY(self.self, self.other * self.alpha)

    @typing.final
    @typing.override
    def cost(self) -> int:
        return self._shape.numel()

    @functools.cached_property
    def _shape(self) -> torch.Size:
        self_is_tensor = isinstance(self.self, torch.Tensor)
        other_is_tensor = isinstance(self.other, torch.Tensor)

        match self_is_tensor, other_is_tensor:
            case True, True:
                return torch.broadcast_shapes(self.self.shape, self.other.shape)
            case True, False:
                return self.self.shape
            case False, True:
                return self.other.shape
            case _, _:
                raise RuntimeError("Not happening.")


class AddTensor(_BinaryUFunc):
    KEY = ops.aten.add.Tensor
    BINARY = operator.add


class AddScalar(_BinaryUFunc):
    KEY = ops.aten.add.Scalar
    BINARY = operator.add


class SubTensor(_BinaryUFunc):
    KEY = ops.aten.sub.Tensor
    BINARY = operator.sub


class SubScalar(_BinaryUFunc):
    KEY = ops.aten.sub.Scalar
    BINARY = operator.sub


class MulTensor(_BinaryUFunc):
    KEY = ops.aten.mul.Tensor
    BINARY = operator.mul


class MulScalar(_BinaryUFunc):
    KEY = ops.aten.mul.Scalar
    BINARY = operator.mul


class TrueDivTensor(_BinaryUFunc):
    KEY = ops.aten.div.Tensor
    BINARY = operator.truediv


class TrueDivScalar(_BinaryUFunc):
    KEY = ops.aten.div.Scalar
    BINARY = operator.truediv


class FloorDiv(_BinaryUFunc):
    KEY = ops.aten.floor_divide.default
    BINARY = operator.floordiv


class EqTensor(_BinaryUFunc):
    KEY = ops.aten.eq.Tensor
    BINARY = operator.eq


class EqScalar(_BinaryUFunc):
    KEY = ops.aten.eq.Scalar
    BINARY = operator.eq


class NeTensor(_BinaryUFunc):
    KEY = ops.aten.ne.Tensor
    BINARY = operator.ne


class NeScalar(_BinaryUFunc):
    KEY = ops.aten.ne.Scalar
    BINARY = operator.ne


class GeTensor(_BinaryUFunc):
    KEY = ops.aten.ge.Tensor
    BINARY = operator.ge


class GeScalar(_BinaryUFunc):
    KEY = ops.aten.ge.Scalar
    BINARY = operator.ge


class GtTensor(_BinaryUFunc):
    KEY = ops.aten.gt.Tensor
    BINARY = operator.gt


class GtScalar(_BinaryUFunc):
    KEY = ops.aten.gt.Scalar
    BINARY = operator.gt


class LeTensor(_BinaryUFunc):
    KEY = ops.aten.le.Tensor
    BINARY = operator.le


class LeScalar(_BinaryUFunc):
    KEY = ops.aten.le.Scalar
    BINARY = operator.le


class LtTensor(_BinaryUFunc):
    KEY = ops.aten.lt.Tensor
    BINARY = operator.lt


class LtScalar(_BinaryUFunc):
    KEY = ops.aten.lt.Scalar
    BINARY = operator.lt
