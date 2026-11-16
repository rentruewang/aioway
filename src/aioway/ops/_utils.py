# Copyright (c) AIoWay Authors - All Rights Reserved

"Utilities shared by `Op`s."

import functools as ft
from collections import abc as cabc

from aioway._pytrees import tree_map_only
from aioway.t import Domain, FloatTensorDomain, IntTensorDomain, TensorDomain

from .ops import Op, OpDomainMismatch

__all__ = [
    "require_op",
    "require_tensor_op",
    "require_float_op",
    "require_int_op",
    "exec_nested_ops",
]


def require_op(
    op: Op, domain: Domain, check: cabc.Callable[[Op], None] = lambda _: None
) -> None:
    """
    Check `op`'s domain against `domain`.

    A sub domain is a match, `check` is then performed on `op`.
    A super domain (e.g. `ANY`) may match, not known yet, so nothing is done.

    Raises:
        OpDomainMismatch: if neither a sub domain nor a super domain.
    """

    actual = op.__domain__

    # Checks are run if the domain is what we wanted.
    if actual <= domain:
        check(op)
        return

    # Super type is broader, give it a pass.
    if actual > domain:
        return

    raise OpDomainMismatch(f"expected {domain}, got {actual}")


require_tensor_op = ft.partial(require_op, domain=TensorDomain())
"Any tensor."

require_float_op = ft.partial(require_op, domain=FloatTensorDomain())
"A float tensor."

require_int_op = ft.partial(require_op, domain=IntTensorDomain())
"An int tensor."


def _exec_op(obj):
    """
    Execute `obj` if it is `Op`, else return identity.
    """

    if isinstance(obj, Op):
        return obj()

    else:
        return obj


def exec_nested_ops(tree):
    """
    Replace every `Op` in `tree` (at any depth) with its evaluated value.

    This uses `tree_map_only`, but ignore the decomposition on `Op`.
    """

    is_op = lambda obj: isinstance(obj, Op)
    return tree_map_only(Op, _exec_op, tree, is_leaf=is_op)
