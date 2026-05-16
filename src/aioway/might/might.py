# Copyright (c) AIoWay Authors - All Rights Reserved

import abc
import dataclasses as dcls
import typing
from collections import abc as cabc

from torch import nn

from aioway._common import dcls_frozen_no_repr
from aioway.op import Op

__all__ = ["Might", "find_might", "all_mights"]


@dcls_frozen_no_repr
class Might(Op[type[nn.Module]], abc.ABC):
    """
    `Might` is a preview of how an `nn.Module` would be initialized.
    It stands for [m]odule [in]it of wei[ght]s. Or the modules you [might] want.

    It provides metadata as to what `nn.Module` arguments are valid or not,
    much like how `Fate`'s objects mimicks the function signature of `torch.ops.aten.*`.
    """

    KEY: typing.ClassVar[type[nn.Module]] = NotImplemented

    @typing.override
    def do(self) -> nn.Module:
        from aioway.fn import NnInitFn

        return NnInitFn(func=self.KEY, args=(), kwargs=dcls.asdict(self)).do()

    @classmethod
    @typing.override
    def name(cls) -> str:
        return "might::" + cls.__name__


def find_might(nn_type: type[nn.Module], *args, **kwargs) -> Might:
    """
    Get a `Might` from the `nn.Module` type. If not found, return `NotImplemented`.
    """

    # Right now, each `Might` should have distinct key, so just return the 1st.
    # Just get the type. If an error is raised, construction failed,
    # pass the error back, since upper level signature failed.
    for might_type in Might.find(nn_type):
        return might_type(*args, **kwargs)

    # No implementation found.
    return NotImplemented


@typing.no_type_check
def all_mights():
    """
    Get the registry for previews.
    """

    return list(_iter_previews(Might))


def _iter_previews(cls: type[Might]) -> cabc.Generator[type[Might]]:
    for sub in cls.__subclasses__():
        if sub.is_concrete():
            yield sub

        yield from _iter_previews(sub)
