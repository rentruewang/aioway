# Copyright (c) AIoWay Authors - All Rights Reserved

import abc
import dataclasses as dcls
import typing

from torch import nn

from aioway._common import dcls_no_repr
from aioway._common.renders import render_fcall
from aioway._keyed import Keyed
from aioway.fn import NnInitFn

__all__ = ["MessInit", "find_might", "all_mights"]


@dcls_no_repr
class MessInit(Keyed[type[nn.Module]], abc.ABC):
    """
    `MessInit` is a preview of how an `nn.Module` would be initialized.
    It is the init part of `Mess`.

    It provides metadata as to what `nn.Module` arguments are valid or not,
    much like how `Fate`'s objects mimicks the function signature of `torch.ops.aten.*`.
    """

    KEY: typing.ClassVar[type[nn.Module]] = NotImplemented

    @typing.override
    def __repr__(self) -> str:
        return render_fcall("might::" + self._name(), **dcls.asdict(self))

    def do(self) -> nn.Module:
        return NnInitFn(func=self.KEY, args=(), kwargs=dcls.asdict(self)).do()


def find_might(nn_type: type[nn.Module], *args, **kwargs) -> MessInit:
    """
    Get a `Might` from the `nn.Module` type. If not found, return `NotImplemented`.
    """

    # Right now, each `Might` should have distinct key, so just return the 1st.
    # Just get the type. If an error is raised, construction failed,
    # pass the error back, since upper level signature failed.
    for might_type in MessInit.find(nn_type):
        return might_type(*args, **kwargs)
    else:
        return NotImplemented


@typing.no_type_check
def all_mights():
    """
    Get the registry for previews.
    """

    return list(MessInit.impls())
