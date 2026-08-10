# Copyright (c) AIoWay Authors - All Rights Reserved

import abc
import contextlib as ctxl
import dataclasses as dcls
import typing
from collections import abc as cabc

from torch import nn
from torchrl import data as rldata

from aioway._utils import AnySet

__all__ = [
    "EmitterLike",
    "Emitter",
    "FuncEmitter",
    "emitter_dcls",
    "emit_one",
    "emit",
    "emitter_function",
    "emitters_in_scope",
]

_EMITTERS: AnySet[Emitter] = AnySet()
"The emitters that are considered."


def emit_one(observ: rldata.TensorSpec, action: rldata.TensorSpec, /) -> nn.Module:
    """
    A convenient wrapper to only emit the first target found.
    """

    return next(emit(observ, action))


def emit(
    observ: rldata.TensorSpec, action: rldata.TensorSpec, /
) -> cabc.Generator[nn.Module]:
    """
    Emit some candidates based on the given spaces.
    """

    for emitter in emitters_in_scope():
        result = emitter(observ, action)

        if result is NotImplemented:
            continue

        yield result


class EmitterLike(typing.Protocol):
    """
    The baseline function that `emit` uses to generate `nn.Module`s.
    If the space is not supported, `NotImplemented` should be returned.

    All the `BaseLine`s are registered, and iterated over during `Emitter` call.

    Right now a `BaseLine` is stateless, perhaps find a way to make it stateful?
    """

    @abc.abstractmethod
    def __call__(
        self, observ: rldata.TensorSpec, action: rldata.TensorSpec, /
    ) -> nn.Module:
        raise NotImplementedError


@typing.dataclass_transform(frozen_default=True)
def emitter_dcls[T](cls: type[T]) -> type[T]:
    return dcls.dataclass(frozen=True)(cls)


@emitter_dcls
class Emitter(EmitterLike, abc.ABC):
    """
    The base class for `Emitter`s.
    """

    @typing.final
    def __post_init__(self) -> None:
        self._validate()

    def _validate(self) -> None:
        """
        Subclass can overwrite this to perform additional checks.
        """

    @ctxl.contextmanager
    def consider(self) -> cabc.Generator[typing.Self]:
        """
        Consider the current instance of emitter.
        """

        _EMITTERS.add(self)
        try:
            yield self
        finally:
            _EMITTERS.discard(self)


@emitter_dcls
class FuncEmitter(Emitter):
    "The emitter function."

    function: EmitterLike
    """
    The function to wrap.
    """

    def __call__(
        self, observ: rldata.TensorSpec, action: rldata.TensorSpec, /
    ) -> nn.Module:
        return self.function(observ, action)


def emitter_function(emitter: EmitterLike) -> FuncEmitter:
    "The decorator function to wrap a function as an `Emitter`."
    return FuncEmitter(emitter)


def emitters_in_scope() -> AnySet[Emitter]:
    "The emitters that are alive and in the scope of consideration."
    return _EMITTERS
