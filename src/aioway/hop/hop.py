# Copyright (c) AIoWay Authors - All Rights Reserved

"The operator base class."

import abc
import dataclasses as dcls
import graphlib
import typing
from collections import abc as cabc

__all__ = ["Hop", "HopDag"]


@typing.dataclass_transform(frozen_default=True, kw_only_default=True)
def hop_dcls(cls):
    return dcls.dataclass(frozen=True, kw_only=True)(cls)


@hop_dcls
class Hop[T = object](abc.ABC):
    """
    `Hop` stands for [h]igh level [op]erator, or [h]igh level [o]peration [p]review.
    It is essentailly an unevaluated expression that supports inspection.
    """

    def __hash__(self):
        return id(self)

    @abc.abstractmethod
    def deps(self) -> cabc.Iterator[Hop]:
        """
        The dependent `Hop`s.
        """

        raise NotImplementedError

    @abc.abstractmethod
    def do(self) -> T:
        """
        Evaluates the current operator and outputs the results.
        The object must be decomposed into pure tensors (no extra items e.g. primitives).
        """

        raise NotImplementedError


@dcls.dataclass(frozen=True)
class HopDag:
    """
    `HopDag` is a DAG of `Hop`s, ordered in the linear sense.
    """

    hops: cabc.Sequence[Hop]
    "The topologically sorted `Hop`s list."

    @classmethod
    def from_output_hops(cls, outputs: cabc.Iterable[Hop]) -> typing.Self:
        visited: set[Hop] = set()
        for hop in outputs:
            _collect_hops_rec(hop, visited)

        # Format `graphlib.TopologicalSorter` expects: `{node: node.inputs}`.
        hop_graph = {hop: list(hop.deps()) for hop in visited}

        topo_sorter = graphlib.TopologicalSorter(hop_graph)
        topo_sorter.prepare()
        ordered_hop_list = list(topo_sorter.static_order())

        return cls(ordered_hop_list)


def _collect_hops_rec(hop: Hop, visited: set[Hop]) -> None:
    if hop in visited:
        return

    visited.add(hop)
    for dep in hop.deps():
        _collect_hops_rec(dep, visited)
