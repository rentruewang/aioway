# Copyright (c) AIoWay Authors - All Rights Reserved

import abc
import contextlib as ctxl
import dataclasses as dcls
import typing
from collections import abc as cabc

import torch
from torch import nn

from aioway._utils import AnySet
from aioway.tspecs import TSpec

__all__ = [
    "EmitterLike",
    "Emitter",
    "FuncEmitter",
    "emitter_dcls",
    "emit_one",
    "emit",
    "emitter_function",
    "emitters_in_scope",
    "sample_from_tspec",
    "set_batch_size",
]

_EMITTERS: AnySet[Emitter] = AnySet()
"The emitters that are considered."

_batch_size: torch.Size | None = None
"The batch size to use for emitting."


@ctxl.contextmanager
def set_batch_size(*batch_size: int) -> cabc.Generator[None]:
    "Configure the batch size to use with `spec` and `emit`."

    global _batch_size
    _batch_size = torch.Size(batch_size)

    try:
        yield
    finally:
        _batch_size = None


def sample_from_tspec(spec: TSpec, /) -> typing.Any:
    "Sample from the `spec` with the batch size configured by `with_batch_size`."
    assert _batch_size
    return spec.sample(torch.Size(_batch_size))


def emit_one(observ: TSpec, action: TSpec) -> nn.Module:
    """
    A convenient wrapper to only emit the first target found.
    """

    return next(emit(observ, action))


def emit(observ: TSpec, action: TSpec, /) -> cabc.Generator[nn.Module]:
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
    If the `TSpec` is not supported, `NotImplemented` should be returned.

    All the `BaseLine`s are registered, and iterated over during `Emitter` call.

    Right now a `BaseLine` is stateless, perhaps find a way to make it stateful?
    """

    @abc.abstractmethod
    def __call__(self, observ: TSpec, action: TSpec, /) -> nn.Module:
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

    def __call__(self, observ: TSpec, action: TSpec, /) -> nn.Module:
        return self.function(observ, action)


def emitter_function(emitter: EmitterLike) -> FuncEmitter:
    "The decorator function to wrap a function as an `Emitter`."
    return FuncEmitter(emitter)


def emitters_in_scope() -> AnySet[Emitter]:
    "The emitters that are alive and in the scope of consideration."
    return _EMITTERS
