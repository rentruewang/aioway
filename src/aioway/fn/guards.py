# Copyright (c) AIoWay Authors - All Rights Reserved

import re
import typing

import torch
from torch import _ops

__all__ = [
    "TensorFilter",
    "is_leaf_has_grad",
    "all_tensors",
    "is_float",
    "is_int",
    "is_bool",
    "is_aten_op",
    "is_prim_op",
]

_ATEN_OPS = re.compile("aten::.+")
_PRIM_OPS = re.compile("prim::.+")


class TensorFilter(typing.Protocol):
    def __call__(self, tensor: torch.Tensor, /) -> bool: ...


def is_leaf_has_grad(t: torch.Tensor) -> bool:
    return t.is_leaf and t.requires_grad


def all_tensors(_: torch.Tensor):
    return True


def is_float(t: torch.Tensor) -> bool:
    return t.dtype.is_floating_point


def is_int(t: torch.Tensor) -> bool:
    return not is_float(t) and not is_bool(t)


def is_bool(t: torch.Tensor) -> bool:
    return t.dtype == torch.bool


def is_sparse(t: torch.Tensor) -> bool:
    return t.is_sparse


def is_aten_op(op: _ops.OpOverload) -> bool:
    return _dispatch_name(op, _ATEN_OPS)


def is_prim_op(op: _ops.OpOverload) -> bool:
    return _dispatch_name(op, _PRIM_OPS)


def _dispatch_name(op: _ops.OpOverload, regex: re.Pattern[str]) -> bool:
    return bool(regex.fullmatch(op.name()))
