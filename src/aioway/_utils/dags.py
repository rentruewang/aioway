# Copyright (c) AIoWay Authors - All Rights Reserved

"A simple DAG that renders itself and converts to `networkx`."

import dataclasses as dcls
import functools
import graphlib
import typing
from collections import abc as cabc

import numpy as np

__all__ = ["Dag", "DagNode"]


@dcls.dataclass(frozen=True, slots=True)
class DagNode[T]:
    """
    `DagNode` is an interface that
    """

    data: T
    """
    The underlying data.
    """

    deps: list[int]
    """
    List of parent indices.
    """


@dcls.dataclass(frozen=True)
class Dag[T: cabc.Hashable]:
    """
    `Dag` is a DAG of `DagNode`s, ordered in the linear sense.

    If calling the constructor directly, the `nodes` must be ordered linearly already.
    Using `from_*` classmethod instead of the constructors would topo sort the data.
    """

    nodes: list[DagNode[T]]
    "The topologically sorted `T`s list."

    def __post_init__(self) -> None:
        self._verify_sorted()

    def __len__(self) -> int:
        return len(self.nodes)

    def __getitem__(self, idx: int) -> DagNode[T]:
        return self.nodes[idx]

    def __iter__(self):
        for node in self.nodes:
            yield node

    def deps_of(self, idx: int) -> list[int]:
        return self.nodes[idx].deps

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
            for dep_idx in node.deps:
                num_inputs[node_idx] += 1
                num_outputs[dep_idx] += 1

        # Make them immutable.
        num_inputs.flags.writeable = False
        num_outputs.flags.writeable = False
        return num_inputs, num_outputs

    def _verify_sorted(self) -> None:
        for idx, node in enumerate(self.nodes):
            for dep_idx in node.deps:
                if dep_idx < idx:
                    continue

                elif dep_idx == idx:
                    raise ValueError(f"Self reference link detected on {idx=}.")

                else:
                    raise ValueError(f"At {idx=}, {dep_idx=} is greater.")

    @classmethod
    def from_graph(cls, graph: cabc.Mapping[T, cabc.Iterable[T]]) -> typing.Self:
        """
        Create a `Dag` from the given `graph`.
        Uses `graphlib.TopologicalSorter` under the hood.
        """

        topo_sorter = graphlib.TopologicalSorter(graph)
        topo_sorter.prepare()
        ordered_list = list(topo_sorter.static_order())

        item_idx = {item: idx for idx, item in enumerate(ordered_list)}
        dag_list: list[DagNode[T]] = []

        for item in ordered_list:
            dag_list.append(
                DagNode(data=item, deps=[item_idx[dep] for dep in graph[item]])
            )

        return cls(dag_list)
