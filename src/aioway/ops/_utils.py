# Copyright (c) AIoWay Authors - All Rights Reserved

"Checks shared by `Op`s."

from collections import abc as cabc

from aioway.t import Domain, FloatTensorDomain, IntTensorDomain, TensorDomain

from .ops import Op, OpDomainMismatch

__all__ = ["require_op", "require_tensor_op", "require_float_op", "require_int_op"]


def require_op(op: Op, domains: cabc.Sequence[type[Domain]], /) -> type[Domain]:
    """
    Check `op`'s domain against `domains`, in order, and return the one matched.

    Like `isinstance(op.__domain__, tuple(domains))`, except only the first is matched
    with its subclasses. The rest are fallbacks, matched exactly:
    placeholders (e.g. `TensorDomain` from `TensorOp`) until the domain is inferred.

    Raises:
        OpDomainMismatch: if nothing matches.
    """

    domain = op.__domain__
    first, *fallbacks = domains

    if isinstance(domain, first):
        return first

    for fallback in fallbacks:
        if type(domain) is fallback:
            return fallback

    names = ", ".join(d.__name__ for d in domains)
    raise OpDomainMismatch(f"expected one of [{names}], got {domain}")


def require_tensor_op(op: Op, /) -> None:
    "Any tensor."
    require_op(op, [TensorDomain])


def require_float_op(op: Op, /) -> None:
    "A float tensor, or a tensor not inferred yet."
    require_op(op, [FloatTensorDomain, TensorDomain])


def require_int_op(op: Op, /) -> None:
    "An int tensor, or a tensor not inferred yet."
    require_op(op, [IntTensorDomain, TensorDomain])
