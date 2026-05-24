# Copyright (c) AIoWay Authors - All Rights Reserved

"The utilties around DAGs."

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


class Dag[T: cabc.Hashable]:
    """
    `Dag` is a DAG of `DagNode`s, ordered in the linear sense.

    Prefer using the `from_*` classmethod instead of the constructors.
    """

    def __init__(self, ordered: cabc.Sequence[DagNode[T]]) -> None:
        self._ordered_list: cabc.Sequence[DagNode[T]] = ordered
        "The topologically sorted `T`s list."

        self._verify_sorted()

    def __len__(self) -> int:
        return len(self._ordered_list)

    @typing.overload
    def __getitem__(self, idx: int) -> T: ...
    @typing.overload
    def __getitem__(self, idx: slice) -> list[T]: ...
    @typing.no_type_check
    def __getitem__(self, idx):
        if isinstance(idx, int):
            return self._ordered_list[idx].item()

        if isinstance(idx, slice):
            sliced = self._ordered_list[idx]
            return [node.item() for node in sliced]

        raise IndexError(f"Unhandled {idx=}.")

    def __iter__(self):
        for node in self._ordered_list:
            yield node.item()

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

        for node_idx, node in enumerate(self._ordered_list):
            for dep in node.deps():
                assert node_idx == self._node_index[node]
                dep_idx = self._node_index[dep]

                num_inputs[node_idx] += 1
                num_outputs[dep_idx] += 1

        # Make them immutable.
        num_inputs.flags.writeable = False
        num_outputs.flags.writeable = False
        return num_inputs, num_outputs

    def _verify_sorted(self):
        item_to_idx = self._node_index

        for idx, node in enumerate(self._ordered_list):
            for dep in node.deps():
                if item_to_idx[dep] < idx:
                    continue

                raise AssertionError("The given array is not sorted.")

    @functools.cached_property
    def _node_index(self):
        return {key: idx for idx, key in enumerate(self._ordered_list)}

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
