# Copyright (c) AIoWay Authors - All Rights Reserved

import abc
import dataclasses as dcls
import typing

if typing.TYPE_CHECKING:
    from .sets import InstrSet

__all__ = ["Query"]


class Query(abc.ABC):
    """
    A query is a subnet generator.
    """

    def __call__(self, dag: InstrSet) -> InstrSet:
        raise NotImplementedError


@dcls.dataclass(frozen=True)
class IndexQuery(Query):
    indices: list[int]

    def __call__(self, dag: InstrSet) -> InstrSet:
        return InstrSet.from_thunk_list([dag[i] for i in self.indices])
