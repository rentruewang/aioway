# Copyright (c) AIoWay Authors - All Rights Reserved

import abc
import dataclasses as dcls
import inspect
import typing
from collections import abc as cabc

from torch import nn

from aioway._iters import Iter
from aioway._utils import dcls_asdict, render_fcall
from aioway.modes import NnInitThunk

from .ufuncs import NnUFunc, NnUFuncThunk

__all__ = ["NnInit", "nn_init_dcls", "find_nn_init", "build_nn_iter"]

_NN_INITS: dict[cabc.Callable[..., nn.Module], type[NnInit]] = {}


@typing.dataclass_transform()
@typing.no_type_check
def nn_init_dcls(cls):
    "Decorator of dataclass for `NnInit`."
    return dcls.dataclass(repr=False)(cls)


@nn_init_dcls
class NnInit(abc.ABC):
    """
    `NnInit` records the signature of an `nn.Module` initialization, and creates it.

    It provides metadata as to what `nn.Module` arguments are valid or not.
    """

    NN: typing.ClassVar[type[nn.Module]] = NotImplemented
    UFUNC: typing.ClassVar[type[NnUFunc]] = NotImplemented

    def __init_subclass__(cls) -> None:
        # Abstract in terms of `ClassVar`.
        if cls.NN is NotImplemented:
            return

        if cls.UFUNC is NotImplemented:
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

    @property
    def ufunc(self) -> NnUFunc:
        nn_module = self.init_nn()
        return self.UFUNC(self, nn_module)

    def init_nn(self) -> nn.Module:
        thunk = NnInitThunk(func=self.NN, **dcls_asdict(self))
        return thunk()

    def apply(self, *args, **kwargs) -> NnUFuncThunk:
        """
        Builds a high level operator with the given `NN` and `HOP` runtime class.
        """

        return self.ufunc.thunk(*args, **kwargs)


def find_nn_init(thunk: NnInitThunk, /) -> NnInit | None:
    """
    Find the `NnInit` type, based on the `nn.Module` type.
    If `NnInit` is not found, `None` is returned.
    """

    if (nn_init_type := _NN_INITS.get(thunk.func)) is None:
        return None

    return nn_init_type(*thunk.args, **thunk.kwargs)


def build_nn_iter(thunk: NnInitThunk, *args, **kwargs) -> Iter | None:
    if (nn_init := find_nn_init(thunk)) is None:
        return None

    return nn_init.apply(*args, **kwargs)
