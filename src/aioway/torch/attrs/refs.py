# Copyright (c) AIoWay Authors - All Rights Reserved

import dataclasses as dcls

from .attrs import Attr


@dcls.dataclass(frozen=True)
class AttrRef:
    "The `Attr` class, with `__id__` of the tensor."

    _: dcls.KW_ONLY

    __id__: int = 0
    "The id of the `Attr`'s fake tensor."

    attr: Attr
    "The attribute that this holds."

    def __getstate__(self):
        """
        The overwritten `__getstate__` of `AttrRef`.

        Since `__hash__` depends on this, we get `__hash__` for free.
        """

        return {"__id__": self.__id__, "attr": self.attr.__getstate__()}
