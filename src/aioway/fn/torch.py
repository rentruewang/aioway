# Copyright (c) AIoWay Authors - All Rights Reserved

import re

from torch import _ops

_ATEN_OPS = re.compile("aten::.+")
_PRIM_OPS = re.compile("prims::.+")

__all__ = ["is_aten_op", "is_prim_op"]


def is_aten_op(op: _ops.OpOverload) -> bool:
    return _dispatch_name(op, _ATEN_OPS)


def is_prim_op(op: _ops.OpOverload) -> bool:
    return _dispatch_name(op, _PRIM_OPS)


def _dispatch_name(op: _ops.OpOverload, regex: re.Pattern) -> bool:
    return bool(_PRIM_OPS.fullmatch(op.name()))
