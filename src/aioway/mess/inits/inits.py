# Copyright (c) AIoWay Authors - All Rights Reserved

import abc
import dataclasses as dcls
import typing

from torch import nn

from aioway._keyed import Keyed
from aioway._types import dcls_no_repr
from aioway.fn import NnInitFn
from aioway.renders import render_fcall

__all__ = ["MessInit"]


@dcls_no_repr
class MessInit(Keyed[type[nn.Module]], abc.ABC):
    """
    `MessInit` is a preview of how an `nn.Module` would be initialized.
    It is the init part of `Mess`.

    It provides metadata as to what `nn.Module` arguments are valid or not.
    """

    KEY: typing.ClassVar[type[nn.Module]] = NotImplemented

    @typing.override
    def __repr__(self) -> str:
        return render_fcall("mess_init::" + self._name(), **dcls.asdict(self))

    def do(self) -> nn.Module:
        return NnInitFn(func=self.KEY, args=(), kwargs=dcls.asdict(self)).do()
