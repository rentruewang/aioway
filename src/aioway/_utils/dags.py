# Copyright (c) AIoWay Authors - All Rights Reserved

"A simple DAG that renders itself and converts to `networkx`."

import abc
import dataclasses as dcls
import functools
import graphlib
import typing
from collections import abc as cabc

import numpy as np

__all__ = ["Dag", "DagNode", "TupleDagNode"]


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
class TupleDagNode[T](DagNode[T]):
    """
    `ListDagNode` is a convenient implementation for `DagNode`.
    """

    data: T
    """
    The data stored by the `DagNode`..
    """

    parents: tuple[DagNode[T], ...]
    """
    The parents (dependencies) of this `ListDagNode`.
    """

    @typing.override
    def item(self) -> T:
        return self.data

    @typing.override
    def deps(self) -> cabc.Iterator[DagNode[T]]:
        yield from self.parents


@dcls.dataclass(frozen=True)
class Dag[T: cabc.Hashable]:
    """
    `Dag` is a DAG of `DagNode`s, ordered in the linear sense.

    If calling the constructor directly, the `nodes` must be ordered linearly already.
    Using `from_*` classmethod instead of the constructors would topo sort the data.
    """

    nodes: cabc.Sequence[DagNode[T]]
    "The topologically sorted `T`s list."

    def __post_init__(self) -> None:
        self._verify_sorted()
        self._verify_unique()

    def __len__(self) -> int:
        return len(self.nodes)

    def __getitem__(self, idx: int) -> DagNode[T]:
        return self.nodes[idx]

    def __iter__(self):
        for node in self.nodes:
            yield node

    @property
    def num_inputs(self):
        i, _ = self._num_inputs_outputs
        return i

    @property
    def num_outputs(self):
        _, o = self._num_inputs_outputs
        return o

    @functools.cached_property
    def _num_inputs_outputs(self):
        num_inputs = np.zeros(len(self), dtype=int)
        num_outputs = np.zeros(len(self), dtype=int)

        for node_idx, node in enumerate(self.nodes):
            for dep in node.deps():
                assert node_idx == self._node_index[node]
                dep_idx = self._node_index[dep]

                num_inputs[node_idx] += 1
                num_outputs[dep_idx] += 1

        # Make them immutable.
        num_inputs.flags.writeable = False
        num_outputs.flags.writeable = False
        return num_inputs, num_outputs

    def _verify_unique(self) -> None:
        values = [node.item() for node in self.nodes]

        if len(values) != len({*values}):
            raise ValueError("The values should be unique!")

    def _verify_sorted(self) -> None:
        node_to_idx = self._node_index

        for idx, node in enumerate(self.nodes):
            for dep in node.deps():
                if node_to_idx[dep] < idx:
                    continue

                raise AssertionError("The given array is not sorted.")

    @functools.cached_property
    def _node_index(self):
        return {key: idx for idx, key in enumerate(self)}

    def networkx(self):
        "Convert the graph to `nx.DiGraph`, using data dependencies as link."

        import networkx as nx

        graph: nx.Graph[T] = nx.Graph()

        for node in self.nodes:
            node_item = node.item()
            graph.add_node(node_item)

            for dep in node.deps():
                dep_item = dep.item()
                _ = graph.add_edge(dep_item, node_item)

        return graph

    @classmethod
    def from_outputs(cls, outputs: cabc.Iterable[DagNode[T]]) -> typing.Self:
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
