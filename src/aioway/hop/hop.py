# Copyright (c) AIoWay Authors - All Rights Reserved

"The [h]igh level [o]peration [p]review class."

import abc
import dataclasses as dcls
import graphlib
import typing
from collections import abc as cabc

from aioway._utils.decomps import decomp_flatten
from aioway.fn import Fn, thunk_dcls

__all__ = ["HopInit", "HopFwd"]


@typing.dataclass_transform(kw_only_default=True)
def hop_init_dcls(cls):
    return dcls.dataclass(match_args=True, kw_only=True)(cls)


@hop_init_dcls
class HopInit(Fn, abc.ABC):
    """
    `HopInit` stands for [h]igh level [o]peration [p]review node.
    It is essentailly an unevaluated expression that supports inspection.
    """

    def __hash__(self) -> int:
        return id(self)

    @typing.final
    def __do__(self) -> HopFwd:
        return self.init()

    @typing.final
    def deps(self) -> cabc.Iterator[HopInit]:
        """
        The dependent `Hop`s. It's decomposed from the dataclass members.
        """

        yield from decomp_flatten(self, HopInit)

    @abc.abstractmethod
    def init(self):
        """
        Perform initialize and output a `HopFwd` object.
        """

        raise NotImplementedError

    @property
    @typing.override
    def types(cls):
        return HopInit


@thunk_dcls
class HopFwd(Fn, abc.ABC):
    """
    `HopFwd` is the node that would be evaluated during run time.
    """

    def __hash__(self):
        return id(self)

    @typing.final
    def __do__(self) -> object:
        return self.fwd()

    @abc.abstractmethod
    def fwd(self) -> object:
        return self.__do__()

    @typing.final
    def deps(self) -> cabc.Iterator[HopFwd]:
        """
        The dependent `Hop`s. It's decomposed from the dataclass members.
        """

        yield from decomp_flatten(self, HopFwd)


@dcls.dataclass
class HopDag[H: HopInit | HopFwd]:
    ordered: list[H]

    @classmethod
    def from_list(cls, items: cabc.Sequence[H]) -> typing.Self:
        """
        Convert from a list of `items`.
        Uses `id` so nodes are all treated as different,
        even if links and data are all equal.
        """

        dag_nodes = []

        ids = {id(item): item for item in items}

        graph: dict[int, list[int]] = {}
        for item in items:
            dep_ids = [id(dep) for dep in item.deps()]
            graph[id(item)] = dep_ids
            assert all(di in ids for di in dep_ids)

        topo_sorter = graphlib.TopologicalSorter(graph)
        topo_sorter.prepare()
        return cls([ids[i] for i in topo_sorter.static_order()])
