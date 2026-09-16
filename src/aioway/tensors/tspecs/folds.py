# Copyright (c) AIoWay Authors - All Rights Reserved

import dataclasses as dcls
import typing
from collections import abc as cabc

from torchrl.data import tensor_specs as tspecs

from aioway._utils import Sign

from .tspecs import TSpec

__all__ = ["FoldTSpec", "tspec_fold_rule"]

_ALL_RULES: dict[FoldSign, FoldRule] = {}


class FoldSign(typing.NamedTuple):
    """
    The signature for `FoldTSpec`.
    This is useful to classify each "rule" s.t. we can make it modular.
    """

    left: type[TSpec]
    "The type of the LHS `TSpec` of the fold."

    right: type[TSpec]
    "The type of the RHS `TSpec` of the fold."


class FoldRule[L: TSpec = typing.Any, R: TSpec = typing.Any](typing.Protocol):
    def __call__(self, left: L, right: R) -> TSpec:
        raise NotImplementedError


def tspec_fold_rule(left: type[TSpec], right: type[TSpec]) -> FoldRule:
    "Get the rule for input `(left, right)`."

    return _ALL_RULES[FoldSign(left, right)]


@dcls.dataclass(frozen=True)
class FoldTSpec(FoldRule):
    """
    The left fold for `TSpec`'s casting.
    This casts the `TSpec`s to a common `TSpec` that contains both.
    """

    tspecs: dict[FoldSign, FoldRule] = dcls.field(default_factory=dict)

    def __call__(self, left: TSpec, right: TSpec) -> TSpec:
        type_left = type(left)
        type_right = type(right)

        sign = FoldSign(type_left, type_right)
        return self.tspecs[sign](left, right)


def register_tspec_fold[F: FoldRule](
    left: type[TSpec], right: type[TSpec], /
) -> cabc.Callable[[F], F]:
    """
    Register a `TSpec` folding rule that has input signature `(left, right)`.
    """

    def decorator(func: F) -> F:
        fold_sign = FoldSign(left, right)
        sign = Sign.from_callable(func)
        params = sign.param_list

        if len(params) != 2:
            raise TypeError(f"{func=} should accept 2 arguments.")
        left_param, right_param = params

        if (annot := left_param.annotation) != left:
            raise TypeError(
                f"Annotation mismatch! {left=}, but function's left signature: {annot}."
            )

        if (annot := right_param.annotation) != right:
            raise TypeError(
                f"Annotation mismatch! {right=}, but function's right signature: {annot}."
            )

        if fold_sign in _ALL_RULES:
            raise KeyError(f"Key {fold_sign} already exists.")

        _ALL_RULES[fold_sign] = func

        return func

    return decorator


@register_tspec_fold(tspecs.Unbounded, tspecs.Unbounded)
def unbounded_and_unbounded(
    left: tspecs.Unbounded, right: tspecs.Unbounded
) -> tspecs.Unbounded:
    left_discrete = isinstance(left, tspecs.UnboundedDiscrete)
    right_discrete = isinstance(right, tspecs.UnboundedDiscrete)

    # Whenever we have int and float casted together, it will be float.
    if left_discrete and right_discrete:
        return tspecs.UnboundedDiscrete()

    else:
        return tspecs.UnboundedContinuous()
