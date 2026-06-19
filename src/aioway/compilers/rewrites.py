# Copyright (c) AIoWay Authors - All Rights Reserved

import typing

from aioway.hop import ListIter

__all__ = ["Rewriter"]


class Rewriter(typing.Protocol):
    """
    `Rewriter` rewrites the given `HopDag` into another `HopDag`.
    """

    def __call__(self, dag: ListIter, /) -> ListIter: ...
