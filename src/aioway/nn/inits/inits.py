# Copyright (c) AIoWay Authors - All Rights Reserved

import abc
import dataclasses as dcls
import inspect
import typing

from torch import nn

from aioway.modes import NnInitFn
from aioway.renders import render_fcall

__all__ = ["NnInit", "find_nn_init"]

_NN_INITS: dict[type[nn.Module], type[NnInit]] = {}


@typing.dataclass_transform(frozen_default=False)
def nn_init_dcls(cls):
    "Decorator of dataclass for `NnInit`."
    return dcls.dataclass(frozen=False, repr=False)(cls)


@nn_init_dcls
class NnInit(abc.ABC):
    """
    `NnInit` records the signature of an `nn.Module` initialization, and creates it.

    It provides metadata as to what `nn.Module` arguments are valid or not.
    """

    NN: typing.ClassVar[type[nn.Module]] = NotImplemented

    def __init_subclass__(cls) -> None:
        # Abstract in terms of `ClassVar`.
        if cls.NN is NotImplemented:
            return

        if inspect.isabstract(cls):
            return

        _NN_INITS[cls.NN] = cls

    @typing.override
    def __repr__(self) -> str:
        return render_fcall("nn_init::" + type(self).__qualname__, **dcls.asdict(self))

    def do(self) -> nn.Module:
        return NnInitFn(func=self.NN, args=(), kwargs=dcls.asdict(self)).do()


def find_nn_init(thunk: NnInitFn, /) -> NnInit | None:
    """
    Find the `NnInit` type, based on the `nn.Module` type.
    If `NnInit` is not found, `None` is returned.
    """

    if (nn_init_type := _NN_INITS.get(thunk.func)) is None:
        return None

    return nn_init_type(*thunk.args, **thunk.kwargs)
