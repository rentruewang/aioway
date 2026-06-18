# Copyright (c) AIoWay Authors - All Rights Reserved

import abc
import functools
import operator
import typing
from collections import abc as cabc

import torch
from torch import ops

from aioway._costs import Cost

from .fate import Aten, aten_dcls

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


@aten_dcls
class _BinaryUFunc(Aten, abc.ABC):
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
    def forward(self) -> torch.Tensor:
        return self.BINARY(self.self, self.other * self.alpha)

    def cost(self) -> Cost:
        numel = self._shape.numel()
        return Cost(time=numel, memory=numel * 2)

    @functools.cached_property
    @typing.no_type_check
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


@aten_dcls
class AddTensor(_BinaryUFunc):
    KEY = ops.aten.add.Tensor
    BINARY = operator.add


@aten_dcls
class AddScalar(_BinaryUFunc):
    KEY = ops.aten.add.Scalar
    BINARY = operator.add


@aten_dcls
class SubTensor(_BinaryUFunc):
    KEY = ops.aten.sub.Tensor
    BINARY = operator.sub


@aten_dcls
class SubScalar(_BinaryUFunc):
    KEY = ops.aten.sub.Scalar
    BINARY = operator.sub


@aten_dcls
class MulTensor(_BinaryUFunc):
    KEY = ops.aten.mul.Tensor
    BINARY = operator.mul


@aten_dcls
class MulScalar(_BinaryUFunc):
    KEY = ops.aten.mul.Scalar
    BINARY = operator.mul


@aten_dcls
class TrueDivTensor(_BinaryUFunc):
    KEY = ops.aten.div.Tensor
    BINARY = operator.truediv


@aten_dcls
class TrueDivScalar(_BinaryUFunc):
    KEY = ops.aten.div.Scalar
    BINARY = operator.truediv


@aten_dcls
class FloorDiv(_BinaryUFunc):
    KEY = ops.aten.floor_divide.default
    BINARY = operator.floordiv


@aten_dcls
class EqTensor(_BinaryUFunc):
    KEY = ops.aten.eq.Tensor
    BINARY = operator.eq


@aten_dcls
class EqScalar(_BinaryUFunc):
    KEY = ops.aten.eq.Scalar
    BINARY = operator.eq


@aten_dcls
class NeTensor(_BinaryUFunc):
    KEY = ops.aten.ne.Tensor
    BINARY = operator.ne


@aten_dcls
class NeScalar(_BinaryUFunc):
    KEY = ops.aten.ne.Scalar
    BINARY = operator.ne


@aten_dcls
class GeTensor(_BinaryUFunc):
    KEY = ops.aten.ge.Tensor
    BINARY = operator.ge


@aten_dcls
class GeScalar(_BinaryUFunc):
    KEY = ops.aten.ge.Scalar
    BINARY = operator.ge


@aten_dcls
class GtTensor(_BinaryUFunc):
    KEY = ops.aten.gt.Tensor
    BINARY = operator.gt


@aten_dcls
class GtScalar(_BinaryUFunc):
    KEY = ops.aten.gt.Scalar
    BINARY = operator.gt


@aten_dcls
class LeTensor(_BinaryUFunc):
    KEY = ops.aten.le.Tensor
    BINARY = operator.le


@aten_dcls
class LeScalar(_BinaryUFunc):
    KEY = ops.aten.le.Scalar
    BINARY = operator.le


@aten_dcls
class LtTensor(_BinaryUFunc):
    KEY = ops.aten.lt.Tensor
    BINARY = operator.lt


@aten_dcls
class LtScalar(_BinaryUFunc):
    KEY = ops.aten.lt.Scalar
    BINARY = operator.lt
