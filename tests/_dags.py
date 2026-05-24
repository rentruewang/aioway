# Copyright (c) AIoWay Authors - All Rights Reserved

"The utilties around DAGs."

import abc
import dataclasses as dcls
import graphlib
import typing
from collections import abc as cabc

__all__ = ["Dag", "DagNode", "ListDagNode"]


class DagNode[T](typing.Protocol):
    """
    `DagNode` is an interface that
    """

    @abc.abstractmethod
    def item(self) -> T:
        "The underlying data that is being stored by this `DagNode`."

        raise NotImplementedError

    @abc.abstractmethod
    def deps(self) -> cabc.Iterator[DagNode[T]]:
        """
        The dependent nodes of this `DagNode`.
        """

        raise NotImplementedError


@dcls.dataclass(frozen=True, slots=True)
class ListDagNode[T](DagNode[T]):
    """
    `ListDagNode` is a convenient implementation for `DagNode`.
    """

    data: T
    """
    The data stored by the `DagNode`..
    """

    parents: list[DagNode[T]]
    """
    The parents (dependencies) of this `ListDagNode`.
    """

    @typing.override
    def item(self) -> T:
        return self.data

    @typing.override
    def deps(self) -> cabc.Iterator[DagNode[T]]:
        yield from self.parents


class Dag[T: cabc.Hashable]:
    """
    `HopDag` is a DAG of `Hop`s, ordered in the linear sense.

    Prefer using the `from_*` classmethod instead of the constructors.
    """

    def __init__(self, ordered: cabc.Sequence[DagNode[T]]) -> None:
        self._ordered_list: cabc.Sequence[DagNode[T]] = ordered
        "The topologically sorted `T`s list."

        self._verify_sorted()

    def __len__(self) -> int:
        return len(self._ordered_list)

    def __getitem__(self, idx: int) -> T:
        return self._ordered_list[idx].item()

    def __iter__(self):
        for node in self._ordered_list:
            yield node.item()

    def _verify_sorted(self):
        item_to_idx = {key: idx for idx, key in enumerate(self)}

        for idx, node in enumerate(self._ordered_list):
            for dep in node.deps():
                if item_to_idx[dep.item()] < idx:
                    continue

                raise AssertionError("The given array is not sorted.")

    @classmethod
    def from_output(cls, outputs: cabc.Iterable[DagNode[T]]) -> typing.Self:
        visited: set[DagNode[T]] = set()
        for node in outputs:
            _collect_dag_nodes_rec(node, visited)

        # Format `graphlib.TopologicalSorter` expects: `{node: node.inputs}`.
        graph = {node: list(node.deps()) for node in visited}

        topo_sorter = graphlib.TopologicalSorter(graph)
        topo_sorter.prepare()
        ordered_list = list(topo_sorter.static_order())

        return cls(ordered_list)


def _collect_dag_nodes_rec[T](node: DagNode[T], visited: set[DagNode[T]]) -> None:
    "Collect the nodes from DAG recursively."

    if node in visited:
        return

    visited.add(node)
    for dep in node.deps():
        _collect_dag_nodes_rec(dep, visited)
