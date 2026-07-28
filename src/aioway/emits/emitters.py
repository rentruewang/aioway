# Copyright (c) AIoWay Authors - All Rights Reserved

import abc
import contextlib as ctxl
import dataclasses as dcls
import typing
from collections import abc as cabc

from aioway._ufuncs import UFunc
from aioway._utils import AnySet
from aioway.spaces import Space

__all__ = [
    "EmitterLike",
    "Emitter",
    "FuncEmitter",
    "emitter_dcls",
    "emit_one",
    "emit",
    "emitters_in_scope",
]

_EMITTERS: AnySet[Emitter] = AnySet()
"The emitters that are considered."


def emit_one(observ_space: Space, action_space: Space) -> UFunc:
    """
    A convenient wrapper to only emit the first target found.
    """

    return next(emit(observ_space=observ_space, action_space=action_space))


def emit(observ_space: Space, action_space: Space) -> cabc.Generator[UFunc]:
    """
    Emit some candidates based on the given spaces.
    """

    for emitter in emitters_in_scope():
        result = emitter(observ_space, action_space)

        if result is not NotImplemented:
            yield result


class EmitterLike(typing.Protocol):
    """
    The baseline function that `emit` uses to generate `UFunc`s.
    If the space is not supported, `NotImplemented` should be returned.

    All the `BaseLine`s are registered, and iterated over during `Emitter` call.

    Right now a `BaseLine` is stateless, perhaps find a way to make it stateful?
    """

    @abc.abstractmethod
    def __call__(self, observation_space: Space, action_space: Space, /) -> UFunc:
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

    def __call__(self, observation_space: Space, action_space: Space) -> UFunc:
        return self.function(observation_space, action_space)


def emitters_in_scope() -> AnySet[Emitter]:
    "The emitters that are alive and in the scope of consideration."
    return _EMITTERS
