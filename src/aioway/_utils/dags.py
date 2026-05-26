# Copyright (c) AIoWay Authors - All Rights Reserved

"Simple dag utilites."

import dataclasses as dcls
import functools
import graphlib
import typing
from collections import abc as cabc

__all__ = ["DagNode", "Dag", "topo_sort"]


class DagNode[T](typing.NamedTuple):
    """
    `DagNode` essentially is a `key: list[value]` mapping pair,
    but `key` does not need to be hashable. Each data must have unique `id`.
    """

    key: T
    """
    The underlying data.
    """

    deps: list[T]
    """
    List of parents.
    """


@dcls.dataclass(frozen=True)
class Dag[T]:
    """
    The sorted results of `topo_sort`.
    """

    nodes: cabc.Sequence[DagNode[T]]
    "The list of nodes after topological sorting."

    @property
    def items(self) -> list[T]:
        "Get the underlying items."

        return [node.key for node in self.nodes]

    def num_inputs(self):
        ins, _ = self._num_ins_outs
        return ins

    def num_outputs(self):
        _, outs = self._num_ins_outs
        return outs

    @functools.cached_property
    def _num_ins_outs(self):
        input_cnt = [0] * len(self.nodes)
        output_cnt = [0] * len(self.nodes)

        for node_idx, node in enumerate(self.nodes):
            for dep in node.deps:
                assert node_idx == self._lookup_node_idx(node.key)
                dep_idx = self._lookup_node_idx(dep)

                input_cnt[node_idx] += 1
                output_cnt[dep_idx] += 1

        return tuple(input_cnt), tuple(output_cnt)

    def _lookup_node_idx(self, key: T) -> int:
        return self._ids_to_idx[id(key)]

    @functools.cached_property
    def _ids_to_idx(self) -> dict[int, int]:
        return {id(node.key): idx for idx, node in enumerate(self.nodes)}


def topo_sort[T](graph: cabc.Sequence[DagNode[T]], /) -> Dag[T]:
    """
    Create a `Dag` from the given `graph`. Uses `TopologicalSorter` under the hood.
    Used this instead of `TopologicalSorter` when data is not `Hashable`.
    """

    deref = _validate_graph_ids(graph)

    sortable_graph = {id(node.key): [id(dep) for dep in node.deps] for node in graph}
    topo_sorter = graphlib.TopologicalSorter(sortable_graph)
    topo_sorter.prepare()
    ordered_ids = list(topo_sorter.static_order())

    return Dag([deref[i] for i in ordered_ids])


def _validate_graph_ids[T](graph: cabc.Sequence[DagNode[T]]) -> dict[int, DagNode[T]]:
    "Validate the graph and return a dictionary used to deref the `id`s."

    # Check if data all have unique ids.
    data_ids = [id(node.key) for node in graph]
    data_ids_set = frozenset(data_ids)

    if len(data_ids_set) != len(data_ids):
        raise ValueError("All data must have unique ids!")

    for node in graph:
        for dep in node.deps:
            if id(dep) not in data_ids_set:
                raise ValueError("Contains dependencies not in the key of `graph`.")

    return {id(node.key): node for node in graph}
