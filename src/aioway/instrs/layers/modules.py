# Copyright (c) AIoWay Authors - All Rights Reserved

import abc
import dataclasses as dcls
import inspect
import typing
from collections import abc as cabc

from torch import nn

from aioway._utils import dcls_asdict, render_fcall

__all__ = ["NnInstr", "nn_instr_dcls"]

_NN_INITS: dict[cabc.Callable[..., nn.Module], type[NnInstr]] = {}


@typing.dataclass_transform()
@typing.no_type_check
def nn_instr_dcls(cls):
    "Decorator of dataclass for `NnInstr`."
    return dcls.dataclass(repr=False)(cls)


@nn_instr_dcls
class NnInstr(abc.ABC):
    """
    `NnInstr` records the signature of an `nn.Module` initialization, and creates it.

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
        return render_fcall("nn_init::" + type(self).__qualname__, **dcls_asdict(self))

    def __call__(self) -> nn.Module:
        return self.NN(**dcls_asdict(self))
