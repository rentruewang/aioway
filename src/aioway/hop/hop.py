# Copyright (c) AIoWay Authors - All Rights Reserved

"The [h]igh level [o]peration [p]review class."

import abc
import dataclasses as dcls
import typing
from collections import abc as cabc

from aioway.fn import NodeFn, thunk_dcls

__all__ = ["HopInit", "HopFwd"]


@typing.dataclass_transform(kw_only_default=True)
def hop_init_dcls(cls):
    return dcls.dataclass(match_args=True, kw_only=True)(cls)


@hop_init_dcls
class HopInit(NodeFn, abc.ABC):
    """
    `HopInit` stands for [h]igh level [o]peration [p]review node.
    It is essentailly an unevaluated expression that supports inspection.
    """

    def __hash__(self):
        return id(self)

    @abc.abstractmethod
    def deps(self) -> cabc.Iterator[HopInit]:
        """
        The dependent `Hop`s.
        """

        raise NotImplementedError

    @abc.abstractmethod
    def __do__(self) -> HopFwd:
        raise NotImplementedError

    def init(self):
        """
        Perform initialize and output a `HopFwd` object.
        """

        return self.__do__()

    @classmethod
    @typing.override
    def _node_type(cls):
        return HopInit


@thunk_dcls
class HopFwd(NodeFn, abc.ABC):
    """
    `HopFwd` is the node that would be evaluated during run time.
    """

    @classmethod
    @typing.override
    def _node_type(cls):
        return HopFwd
