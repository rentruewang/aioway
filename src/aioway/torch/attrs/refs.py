# Copyright (c) AIoWay Authors - All Rights Reserved

import abc
import dataclasses as dcls

from .attrs import Attr, attr_dcls


@dcls.dataclass(frozen=True)
class Ref(abc.ABC):
    __id__: int = 0
    "The id of the `Attr`'s fake tensor."


@attr_dcls
class AttrRef(Ref, Attr):
    "The `Attr` class, with `__id__` of the tensor."

    def __getstate__(self):
        """
        The overwritten `__getstate__` of `AttrRef`.

        Since `__hash__` depends on this, we get `__hash__` for free.
        """

        mapping = super().__getstate__()
        mapping["__id__"] = self.__id__
        return mapping
