# Copyright (c) AIoWay Authors - All Rights Reserved

"The [h]igh level [o]peration [p]review class."

import abc
import dataclasses as dcls
import typing
from collections import abc as cabc

from aioway._fn import Fn, thunk_dcls
from aioway._utils.decomps import decomp_flatten

__all__ = ["HopInit", "HopFwd"]


@typing.dataclass_transform(kw_only_default=True)
def hop_init_dcls(cls):
    return dcls.dataclass(match_args=True, kw_only=True)(cls)


@hop_init_dcls
class HopInit(abc.ABC):
    """
    `HopInit` stands for [h]igh level [o]peration [p]review node.
    It is essentailly an unevaluated expression that supports inspection.
    """

    def __hash__(self) -> int:
        return id(self)

    @typing.final
    def __call__(self) -> HopFwd:
        return self.init()

    @abc.abstractmethod
    def init(self) -> HopFwd:
        raise NotImplementedError

    @typing.final
    def deps(self) -> cabc.Iterator[HopInit]:
        """
        The dependent `Hop`s. It's decomposed from the dataclass members.
        """

        yield from decomp_flatten(self, HopInit)


@thunk_dcls
class HopFwd(Fn, abc.ABC):
    """
    `HopFwd` is the node that would be evaluated during run time.
    """

    def __hash__(self):
        return id(self)

    @typing.final
    def __call__(self) -> object:
        return self.forward()

    @abc.abstractmethod
    def forward(self) -> object:
        raise NotImplementedError

    @typing.final
    def deps(self) -> cabc.Iterator[HopFwd]:
        """
        The dependent `Hop`s. It's decomposed from the dataclass members.
        """

        yield from decomp_flatten(self, HopFwd)
