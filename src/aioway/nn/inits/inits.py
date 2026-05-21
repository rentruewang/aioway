# Copyright (c) AIoWay Authors - All Rights Reserved

import abc
import dataclasses as dcls
import typing

from torch import nn

from aioway.modes import NnInitFn
from aioway.renders import render_fcall

__all__ = ["NnInit"]


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

    @typing.override
    def __repr__(self) -> str:
        return render_fcall("nn_init::" + type(self).__qualname__, **dcls.asdict(self))

    def do(self) -> nn.Module:
        return NnInitFn(func=self.NN, args=(), kwargs=dcls.asdict(self)).do()
