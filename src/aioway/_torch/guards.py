# Copyright (c) AIoWay Authors - All Rights Reserved

import abc
import re
import typing
from collections import abc as cabc

import torch
from torch import _ops

__all__ = [
    "HasParam",
    "TensorFilter",
    "is_leaf_has_grad",
    "filter_tensor_off",
    "is_float_tensor",
    "is_int_tensor",
    "is_bool_tensor",
    "is_aten_op",
    "is_prim_op",
    "is_torchvision_op",
    "is_torchcodec_op",
]

_ATEN_OPS = re.compile("aten::.+")
_PRIM_OPS = re.compile("prim::.+")
_TORCHVISION_OPS = re.compile("image::.+")
_TORCHCODEC_OPS = re.compile("torchcodec_ns::.+")


class TensorFilter(typing.Protocol):
    def __call__(self, tensor: torch.Tensor, /) -> bool: ...


def is_leaf_has_grad(t: torch.Tensor) -> bool:
    return t.is_leaf and t.requires_grad


def filter_tensor_off(_: torch.Tensor):
    return True


def is_float_tensor(t: torch.Tensor) -> bool:
    return t.dtype.is_floating_point


def is_int_tensor(t: torch.Tensor) -> bool:
    return not is_float_tensor(t) and not is_bool_tensor(t)


def is_bool_tensor(t: torch.Tensor) -> bool:
    return t.dtype == torch.bool


def is_sparse_tensor(t: torch.Tensor) -> bool:
    return t.is_sparse


def is_aten_op(op: _ops.OpOverload) -> bool:
    return _dispatch_name(op, _ATEN_OPS)


def is_prim_op(op: _ops.OpOverload) -> bool:
    return _dispatch_name(op, _PRIM_OPS)


def is_torchvision_op(op: _ops.OpOverload) -> bool:
    return _dispatch_name(op, _TORCHVISION_OPS)


def is_torchcodec_op(op: _ops.OpOverload) -> bool:
    return _dispatch_name(op, _TORCHCODEC_OPS)


def _dispatch_name(op: _ops.OpOverload, regex: re.Pattern[str]) -> bool:
    return bool(regex.fullmatch(op.name()))


class HasParam(abc.ABC):
    """
    `HasParam` is a mixin that requires you to implement `tensors`,
    providing `parameters(select)` which iterates over the tensors and filter them.
    """

    def parameters(self, select: TensorFilter = filter_tensor_off, /):
        """
        Calls `.tensors()` and then use `select` to iterate over the tensors.
        """

        for tensor in self.tensors():
            if select(tensor):
                yield tensor

    @abc.abstractmethod
    def tensors(self) -> cabc.Iterator[torch.Tensor]:
        """
        All the tensors that this `HasParam` uses.
        """

        raise NotImplementedError
