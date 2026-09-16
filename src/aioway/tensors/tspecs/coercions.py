# Copyright (c) AIoWay Authors - All Rights Reserved

import dataclasses as dcls
import logging
import typing
from collections import abc as cabc

from torchrl.data import tensor_specs as tspecs

from aioway._utils import Sign

from .tspecs import TSpec

__all__ = ["TSpecCoercion", "tspec_coercion_rule", "default_coercion", "default_coerce"]

_COERCION_RULES: dict[CoerceSign, CoerceRule] = {}
LOGGER = logging.getLogger(__name__)


class CoerceSign(typing.NamedTuple):
    """
    The signature for `FoldTSpec`.
    This is useful to classify each "rule" s.t. we can make it modular.
    """

    left: type[TSpec]
    "The type of the LHS `TSpec` of the fold."

    right: type[TSpec]
    "The type of the RHS `TSpec` of the fold."


class CoerceRule[L: TSpec = typing.Any, R: TSpec = typing.Any](typing.Protocol):
    def __call__(self, left: L, right: R) -> TSpec:
        raise NotImplementedError


def tspec_coercion_rule(left: type[TSpec], right: type[TSpec]) -> CoerceRule:
    "Get the rule for input `(left, right)`."

    return _COERCION_RULES[CoerceSign(left, right)]


def default_coercion() -> TSpecCoercion:
    "Get the default coercion rule."

    return TSpecCoercion(_COERCION_RULES)


def default_coerce(left: TSpec, right: TSpec) -> TSpec:
    "Coerce with the default rule."

    return default_coercion()(left, right)


@dcls.dataclass(frozen=True)
class TSpecCoercion(CoerceRule):
    """
    The coercion rule collection for `TSpec`, given 2 tspecs.
    """

    tspecs: dict[CoerceSign, CoerceRule] = dcls.field(default_factory=dict)
    "The tspecs that are registered."

    strict: bool = True
    """
    If strict, `NotImplemented` would not be discarded, but propagated.

    This means that if either value is `NotImplemented`, `NotImplemented` is returned.
    """

    def __contains__(self, sign: CoerceSign) -> bool:
        return sign in self.tspecs

    def __len__(self) -> int:
        return len(self.tspecs)

    def __getitem__(self, key: CoerceSign) -> CoerceRule:
        return self.tspecs[key]

    def __iter__(self) -> cabc.Generator[CoerceSign]:
        yield from self.tspecs

    def __call__(self, left: TSpec, right: TSpec) -> TSpec:
        if self.strict and (left is NotImplemented or right is NotImplemented):
            return NotImplemented

        elif left is NotImplemented:
            return right

        elif right is NotImplemented:
            return left

        for [lt, rt], impl in self.tspecs.items():
            LOGGER.debug("Attempting to match %s, %s", lt, rt)

            if isinstance(left, lt) and isinstance(right, rt):
                LOGGER.debug("Matched. Implementation: %r", impl)
                return impl(left, right)

        LOGGER.debug("Failed to match.")
        failed = CoerceSign(type(left), type(right))
        raise KeyError(f"No matching implementation found for {failed}.")


def register_tspec_fold[F: CoerceRule](
    left: type[TSpec], right: type[TSpec], /
) -> cabc.Callable[[F], F]:
    """
    Register a `TSpec` folding rule that has input signature `(left, right)`.
    """

    def decorator(func: F) -> F:
        fold_sign = CoerceSign(left, right)
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

        if fold_sign in _COERCION_RULES:
            raise KeyError(f"Key {fold_sign} already exists.")

        _COERCION_RULES[fold_sign] = func

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
