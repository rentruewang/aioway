# Copyright (c) AIoWay Authors - All Rights Reserved

import abc
import dataclasses as dcls
import inspect
import typing

from torch import nn

from aioway._fn import Fn, thunk_dcls
from aioway._utils import render_fcall
from aioway.modes import NnInitFn

from ..hop import HopFwd, HopInit, hop_init_dcls

__all__ = ["NnInit", "find_nn_init", "build_nn_hop", "NnHopInit", "NnHopFwd"]

_NN_INITS: dict[type[nn.Module], type[NnInit]] = {}


@typing.dataclass_transform(frozen_default=False)
def nn_init_dcls(cls):
    "Decorator of dataclass for `NnInit`."
    return dcls.dataclass(frozen=False, repr=False)(cls)


@nn_init_dcls
class NnInit(Fn, abc.ABC):
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

    @abc.abstractmethod
    def __call__(self) -> nn.Module:
        return NnInitFn(func=self.NN, args=(), kwargs=dcls.asdict(self)).init()

    def apply_hop(self, input: HopInit):
        """
        Apply the `NnInit` on the input `Hop` operator.
        Create an `Fn` that when called `.init()`, will initialize an `nn.Module`.

        Returns:
            An `NnHopInit` instance that uses the `input` as input and `self` as module.
        """

        return NnHopInit(nn_init=self, input=input)


def find_nn_init(thunk: NnInitFn, /) -> NnInit | None:
    """
    Find the `NnInit` type, based on the `nn.Module` type.
    If `NnInit` is not found, `None` is returned.
    """

    if (nn_init_type := _NN_INITS.get(thunk.func)) is None:
        return None

    return nn_init_type(*thunk.args, **thunk.kwargs)


def build_nn_hop(thunk: NnInitFn, input: HopInit) -> HopInit | None:
    """
    Build a high level operator from the `thunk` with `input` as input.
    """

    if (nn_init := find_nn_init(thunk)) is None:
        return None

    return nn_init.apply_hop(input)


@hop_init_dcls
class NnHopInit(HopInit):
    """
    The `nn.Module` high level operator.
    """

    nn_init: NnInit
    "The `nn.Module` instance that takes in `input.init()` as input."

    input: HopInit
    "The input `Hop`, must output in a way that `module` accepts."

    def init(self) -> NnHopFwd:
        "Pass the input to the module and returns the output."

        # Initialize here, cost will be tracked outside.
        module = self.nn_init.init()

        return NnHopFwd(module, self.input.init())


@thunk_dcls
class NnHopFwd(HopFwd):
    """
    The `HopFwdNode` subclass for `nn.Module`s.
    It is a thunk so it has args, kwargs as attributes.
    """

    func: nn.Module
    "`NnHopFwd` stores the module."

    args: tuple[typing.Any, ...]
    "The *args arguments."

    kwargs: dict[str, typing.Any]
    "The **kwargs arguments."

    def __init__(self, func: nn.Module, *args, **kwargs):
        super().__init__()

        self.func = func
        self.args = args
        self.kwargs = kwargs

    @typing.override
    def __call__(self) -> object:
        def maybe_do(item):
            if isinstance(item, Fn):
                return item()
            else:
                return item

        args = [maybe_do(arg) for arg in self.args]
        kwargs = {key: maybe_do(arg) for key, arg in self.kwargs.items()}
        return self.func(*args, **kwargs)

    def parameters(self):
        "Pass forward the `.parameters()` of modules."
        yield from self.func.parameters()
