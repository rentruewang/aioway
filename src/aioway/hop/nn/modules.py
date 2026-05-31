# Copyright (c) AIoWay Authors - All Rights Reserved

import abc
import dataclasses as dcls
import inspect
import typing
from collections import abc as cabc

from torch import nn

from aioway._utils import dcls_asdict, render_fcall
from aioway.modes import NnInitFn

from ..hop import Hop
from .hop import NnHop

__all__ = ["NnInit", "find_nn_init", "build_nn_hop"]

_NN_INITS: dict[cabc.Callable[..., nn.Module], type[NnInit]] = {}


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

    NN: typing.ClassVar[cabc.Callable[..., nn.Module]] = NotImplemented
    HOP: typing.ClassVar[cabc.Callable[..., NnHop]] = NotImplemented

    def __init_subclass__(cls) -> None:
        # Abstract in terms of `ClassVar`.
        if cls.NN is NotImplemented:
            return

        if cls.HOP is NotImplemented:
            raise RuntimeError(f"{cls=}.HOP is not configured.")

        if inspect.isabstract(cls):
            return

        _NN_INITS[cls.NN] = cls

    @typing.override
    def __repr__(self) -> str:
        return render_fcall("nn_init::" + type(self).__qualname__, **dcls_asdict(self))

    @typing.final
    def __call__(self) -> nn.Module:
        return self.init_nn()

    def init_nn(self) -> nn.Module:
        thunk = NnInitFn(func=self.NN, **dcls_asdict(self))
        return thunk()

    def apply(self, *args, **kwargs) -> Hop:
        """
        Builds a high level operator with the given `NN` and `HOP` runtime class.
        """

        nn_module = self.init_nn()
        return self.HOP(self, nn_module, *args, **kwargs)


def find_nn_init(thunk: NnInitFn, /) -> NnInit | None:
    """
    Find the `NnInit` type, based on the `nn.Module` type.
    If `NnInit` is not found, `None` is returned.
    """

    if (nn_init_type := _NN_INITS.get(thunk.func)) is None:
        return None

    return nn_init_type(*thunk.args, **thunk.kwargs)


def build_nn_hop(thunk: NnInitFn, *args, **kwargs) -> Hop | None:
    if (nn_init := find_nn_init(thunk)) is None:
        return None

    return nn_init.apply(*args, **kwargs)
