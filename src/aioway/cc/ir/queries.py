# Copyright (c) AIoWay Authors - All Rights Reserved

import dataclasses as dcls
import typing

from .sets import Dag

__all__ = ["Query"]


class Query(typing.Protocol):
    """
    A query is a subnet generator.
    """

    def __call__(self, dag: Dag) -> Dag:
        raise NotImplementedError


@dcls.dataclass(frozen=True)
class IndexQuery(Query):
    indices: list[int]

    def __call__(self, dag: Dag) -> Dag:
        return Dag.from_thunk_list([dag[i] for i in self.indices])
