# Copyright (c) AIoWay Authors - All Rights Reserved

import abc
import dataclasses as dcls
import typing

from torch import nn

from aioway._common._types import dcls_no_repr
from aioway._keyed import Keyed
from aioway.fn import NnFwdFn

__all__ = ["Mess"]


@dcls_no_repr
class Mess(Keyed[type[nn.Module]], abc.ABC):
    """
    `Mess` is the runtime information for `nn.Module`,
    containing information of `nn.Module.forward`.

    `Mess` stands for [m]odule [e]xecution [s]ignature [s]ystem.
    It is to `nn.Module.forward` the equivalent of `Might` to `nn.Module.__init__`.
    """

    KEY: typing.ClassVar[type[nn.Module]] = NotImplemented

    def do(self) -> object:
        return NnFwdFn(func=self.KEY, args=(), kwargs=dcls.asdict(self)).do()


@typing.no_type_check
def all_messes():
    """
    Get the registry for `Mess`es.
    """

    return list(Might.impls())
