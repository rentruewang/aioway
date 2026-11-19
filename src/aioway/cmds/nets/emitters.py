# Copyright (c) AIoWay Authors - All Rights Reserved

import abc
import contextlib as ctxl
import dataclasses as dcls
import typing
from collections import abc as cabc

from aioway._signs import Sign
from aioway._utils import AnySet, any_set
from aioway.ops import OpFunc
from aioway.t import Domain

__all__ = [
    "emitter_dcls",
    "EmitterLike",
    "Emitter",
    "RouteEmitter",
    "emit_one",
    "emit",
    "emitter_function",
    "emitters_in_scope",
]

_EMITTERS: AnySet[Emitter] = any_set()
"The emitters that are considered."


def emit_one(observ: Domain, action: Domain) -> OpFunc:
    """
    A convenient wrapper to only emit the first target found.
    """

    route_emitter = RouteEmitter.current_in_scope()
    return route_emitter(observ, action)


def emit(observ: Domain, action: Domain, /) -> cabc.Generator[OpFunc]:
    """
    Emit some candidates based on the given spaces.
    """

    route_emitter = RouteEmitter.current_in_scope()
    yield from route_emitter.candidates(observ, action)


class EmitterLike(typing.Protocol):
    """
    The baseline function that `emit` uses to generate `OpFunc`s.
    If the `Domain`s are not supported, `NotImplemented` should be returned.
    """

    @abc.abstractmethod
    def __call__(self, observ: Domain, action: Domain, /) -> OpFunc:
        raise NotImplementedError


@typing.dataclass_transform(frozen_default=True)
@typing.no_type_check
def emitter_dcls[T](cls: type[T]) -> type[T]:
    return dcls.dataclass(frozen=True)(cls)


@emitter_dcls
class Emitter[O: Domain, A: Domain](EmitterLike, abc.ABC):
    """
    The base class for `Emitter`s.

    `emit`'s annotations are the domains it takes,
    other domains get `NotImplemented` without calling `emit`.
    """

    @typing.final
    def __post_init__(self) -> None:
        self._validate()

    @typing.final
    def __call__(self, observ: Domain, action: Domain, /) -> OpFunc:
        if not isinstance(observ, self.observ_type):
            return NotImplemented

        if not isinstance(action, self.action_type):
            return NotImplemented

        return self.emit(observ, action)

    @property
    def observ_type(self) -> type[O]:
        "The observation domain taken, from the annotation."

        observ_type, _ = self._sign().args_annot().values()
        return observ_type

    @property
    def action_type(self) -> type[A]:
        "The action domain taken, from the annotation."

        _, action_type = self._sign().args_annot().values()
        return action_type

    @abc.abstractmethod
    def emit(self, observ: O, action: A, /) -> OpFunc:
        """
        Emit for the domains in the annotations.
        Return `NotImplemented` if they are still not supported.
        """

        raise NotImplementedError

    def _sign(self) -> Sign:
        "The signature whose annotations are the domains taken."

        return Sign.from_callable(self.emit)

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
class RouteEmitter(Emitter[Domain, Domain]):
    """
    The emitter that explicitly emits from a list of emitters.
    """

    emitters: cabc.Iterable[EmitterLike]
    """
    The emitters to choose to emit from.
    """

    def emit(self, observ: Domain, action: Domain, /) -> OpFunc:
        for func in self.candidates(observ, action):
            return func

        else:
            raise ValueError(f"No candidate found for {observ=} and {action=}.")

    def candidates(self, observ: Domain, action: Domain) -> cabc.Generator[OpFunc]:
        """
        Emit all possible candidates.
        """

        for emitter in self.emitters:
            result = emitter(observ, action)

            if result is NotImplemented:
                continue

            yield result

    @classmethod
    def current_in_scope(cls) -> typing.Self:
        return cls(emitters_in_scope())


@emitter_dcls
class _FuncEmitter[O: Domain, A: Domain](Emitter[O, A]):
    "The emitter function."

    function: cabc.Callable[[O, A], OpFunc]
    """
    The function to wrap, its annotations are the domains taken.
    """

    def emit(self, observ: O, action: A, /) -> OpFunc:
        return self.function(observ, action)

    def _sign(self) -> Sign:
        return Sign.from_callable(self.function)


def emitter_function[O: Domain, A: Domain](
    function: cabc.Callable[[O, A], OpFunc],
) -> Emitter[O, A]:
    "The decorator function to wrap a function as an `Emitter`."
    return _FuncEmitter(function)


def emitters_in_scope() -> AnySet[Emitter]:
    "The emitters that are alive and in the scope of consideration."
    return _EMITTERS
