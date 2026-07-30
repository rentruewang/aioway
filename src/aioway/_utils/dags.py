# Copyright (c) AIoWay Authors - All Rights Reserved

"Simple dag utilites."

import graphlib
import typing
from collections import abc as cabc

import torch

__all__ = ["TensorInput", "DagNodeKey", "dag_node_key", "topo_sort"]

type DagLike[T] = cabc.Iterable[DagNodeKeyLike[T]] | dict[T, cabc.Iterable[T]]


@typing.runtime_checkable
class TensorInput(typing.Protocol):
    """
    `TensorInput` marks a class whose value depend on input tensors for computation.
    """

    def inputs(self) -> cabc.Iterable[torch.Tensor]:
        "The tensor operands (inputs to the function)"

        raise NotImplementedError


@typing.runtime_checkable
class DagNode(typing.Protocol):
    def deps(self) -> cabc.Iterator[typing.Self]: ...


type DagNodeKeyLike[T] = DagNodeKey[T] | tuple[T, cabc.Iterable[T]]
"A type that looks like `DagNodeKey`."


class DagNodeKey[T](typing.NamedTuple):
    """
    `DagNode` essentially is a `key: list[value]` mapping pair,
    but `key` does not need to be hashable. Each data must have unique `id`.
    """

    key: T
    """
    The underlying data.
    """

    deps: cabc.Sequence[T]
    """
    List of parents.
    """

    @classmethod
    def coerce(cls, item: DagNodeKeyLike[T]) -> typing.Self:
        key, deps = item
        return cls(key=key, deps=list(deps))


def _as_dag_nodes[T](
    obj: cabc.Iterable[DagNodeKeyLike[T]] | dict[T, cabc.Iterable[T]],
) -> cabc.Generator[DagNodeKey[T]]:
    if isinstance(obj, cabc.Mapping):
        obj = typing.cast(cabc.Iterable[DagNodeKeyLike[T]], obj.items())

    for key, deps in obj:
        yield DagNodeKey(key=key, deps=list(deps))


def graph_to_dag_nodes[T](graph: DagLike[T], /) -> list[DagNodeKey[T]]:
    "Convert the `graph` to a `list[DagNodeKey[T]]`."

    return list(_as_dag_nodes(graph))


def dag_node_key(node: DagNode) -> DagNodeKey:
    if not isinstance(node, DagNode):
        raise TypeError(f"{type(node)=} does not conform to the `DagNode` API.")

    return DagNodeKey(key=node, deps=list(node.deps()))


def topo_sort[T](graph: DagLike[T], /) -> list[T]:
    """
    Create a topological sorted list from the given `graph`.
    Uses `TopologicalSorter` under the hood.
    Used this instead of `TopologicalSorter` when data is not `Hashable`.
    """

    node_list: list[DagNodeKey[T]] = graph_to_dag_nodes(graph)
    deref = _validate_graph_ids(node_list)

    sortable_graph = {
        id(node.key): [id(dep) for dep in node.deps] for node in node_list
    }

    # Sort on the ids.
    topo_sorter = graphlib.TopologicalSorter(sortable_graph)
    topo_sorter.prepare()
    ordered_ids = list(topo_sorter.static_order())

    return [deref[i].key for i in ordered_ids]


def _validate_graph_ids[T](
    graph: cabc.Sequence[DagNodeKey[T]],
) -> dict[int, DagNodeKey[T]]:
    "Validate the graph and return a dictionary used to deref the `id`s."

    # Check if data all have unique ids.
    data_ids = [id(node.key) for node in graph]
    data_ids_set = frozenset(data_ids)

    if len(data_ids_set) != len(data_ids):
        raise ValueError("All data must have unique ids!")

    for node in graph:
        for dep in node.deps:
            if id(dep) not in data_ids_set:
                raise ValueError(
                    "List of nodes is not comprehensive! "
                    "Contains dependencies not in the key of `graph`."
                )

    return {id(node.key): node for node in graph}
