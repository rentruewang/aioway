# Copyright (c) AIoWay Authors - All Rights Reserved

"Checks shared by `Op`s."

from aioway.t import Domain, FloatTensorDomain, IntTensorDomain, TensorDomain

from .ops import Op, OpDomainMismatch

__all__ = ["require_op", "require_tensor_op", "require_float_op", "require_int_op"]


def require_op(op: Op, domain: Domain, /) -> None:
    """
    Check `op`'s domain is a sub domain or a super domain of `domain`.

    Raises:
        OpDomainMismatch: if neither.
    """

    actual = op.__domain__

    # A sub domain, e.g. a float tensor for a tensor.
    if actual <= domain:
        return

    # A super domain, e.g. `ANY` or a tensor not inferred yet, may still hold it.
    if actual >= domain:
        return

    raise OpDomainMismatch(f"expected {domain}, got {actual}")


def require_tensor_op(op: Op, /) -> None:
    "Any tensor."
    require_op(op, TensorDomain())


def require_float_op(op: Op, /) -> None:
    "A float tensor, or a tensor not inferred yet."
    require_op(op, FloatTensorDomain())


def require_int_op(op: Op, /) -> None:
    "An int tensor, or a tensor not inferred yet."
    require_op(op, IntTensorDomain())
