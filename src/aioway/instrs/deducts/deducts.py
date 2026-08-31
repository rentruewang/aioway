# Copyright (c) AIoWay Authors - All Rights Reserved

"The deductor type."

import dataclasses as dcls
import functools
import logging
import typing
from collections import abc as cabc

from torch import nn
from torchrl.data import tensor_specs as tspecs

from aioway._utils import Sign
from aioway.tspecs import TSpec, TSpecLike, as_tspec, is_tspec_subtype

__all__ = ["Deductor", "DeductorLike", "DeductorCompat", "deductor_for"]

LOGGER = logging.getLogger(__name__)

type DeductorLike = Deductor | DeductorCompat
"""
Types compatible with `Deductor`.
"""

_DEDUCTOR_REGISTRY: dict[type[nn.Module], Deductor] = {}
"The deductor registry."


@dcls.dataclass(frozen=True)
class DeductorRule:
    "The deductor rules. It wraps a function, whose type hints are all `TSpec`s."

    function: cabc.Callable

    def __post_init__(self) -> None:
        self._validate_deduct_annotations()

    @functools.cached_property
    def signature(self) -> Sign:
        return Sign.from_callable(self.function)

    def _validate_deduct_annotations(self) -> None:
        sign = self.signature
        if not sign.returns_any_type and not is_tspec_subtype(sign.return_annotation):
            raise TypeError(f"{sign=}'s return annotation is not `TSpecLike`.")

        for param in sign.param_list:
            if not param.is_any_type and not is_tspec_subtype(param.annotation):
                raise TypeError(
                    f"{param.annotation=} but it should be a `TSpecLike` type."
                )


class Deductor:
    """
    `Deductor` converts from an input `TSpec` to another `TSpec`.

    It's the type of callables that consumes a torch object and outputs another one.
    """

    def __init__(self, nn_type: type[nn.Module], *impls: cabc.Callable) -> None:
        self._nn_type = nn_type
        self._impls: list[DeductorRule] = []

        for impl in impls:
            self.register(impl)

    def __repr__(self) -> str:
        return f"Deductor({self._nn_type.__name__})"

    def __call__(
        self, *args: TSpecLike, **kwargs: TSpecLike
    ) -> TSpec | tspecs.TensorSpec:
        args_list = [as_tspec(tspec) for tspec in args]
        kwargs_dict = {key: as_tspec(tspec) for key, tspec in kwargs.items()}

        for impl in self._impls:
            result = _attempt_call(impl, *args_list, **kwargs_dict)

            if result is NotImplemented:
                LOGGER.debug("%s failed to parse (*%s, **%s)", impl, args, kwargs)
                continue

            LOGGER.debug("%s successfully parsed (*%s, **%s)", impl, args, kwargs)
            return result
        return NotImplemented

    def register(self, impl: cabc.Callable):
        rule = DeductorRule(impl)
        self._validate_impl(rule)
        self._impls.append(DeductorRule(impl))

    def _validate_impl(self, impl: DeductorRule):
        nn_module_sign = self._nn_module_forward.strip_type()

        if impl.signature.strip_type() != nn_module_sign:
            raise TypeError(
                f"{impl.signature} is not compatible with {nn_module_sign}."
            )

    @functools.cached_property
    def _nn_module_forward(self) -> Sign:
        return Sign.from_callable(self._nn_type.forward).drop_first()


def _attempt_call(impl: cabc.Callable, *args, **kwargs):
    # If signature does not match, don't even attempt.
    if not _signature_matches(impl, *args, **kwargs):
        return NotImplemented

    # If the function itself returns `NotImplemented`, give up.
    if (result := impl(*args, **kwargs)) is NotImplemented:
        return NotImplemented

    return result


def _signature_matches(impl: cabc.Callable, *args, **kwargs) -> bool:
    "Check if signature does match."

    impl_sign = Sign.from_callable(impl)
    arguments = impl_sign.apply(*args, **kwargs)
    params = impl_sign.params
    assert arguments.keys() == params.keys()

    # The arguments must match the type hints.
    for key, typ in params.items():
        if typ.is_any_type:
            continue

        if not isinstance(arguments[key], typ.annotation):
            return False

    return True


@typing.runtime_checkable
class DeductorCompat(typing.Protocol):
    """
    `DeductorCompat` can be converted to a `Deductor`.
    """

    def __deduct__(self) -> Deductor: ...


def deductor_for(nn_type: type[nn.Module]) -> Deductor:
    """
    Get the deductor registered for type of `nn.Module`.
    """

    if nn_type not in _DEDUCTOR_REGISTRY:
        _DEDUCTOR_REGISTRY[nn_type] = Deductor(nn_type)

    return _DEDUCTOR_REGISTRY[nn_type]
