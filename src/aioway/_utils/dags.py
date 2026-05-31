# Copyright (c) AIoWay Authors - All Rights Reserved

"Simple dag utilites."

import graphlib
import typing
from collections import abc as cabc

__all__ = ["DagNode", "topo_sort"]


type DagNodeLike[T] = DagNode[T] | tuple[T, cabc.Iterable[T]]
"A type that looks like `DagNode`."


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

    @classmethod
    def coerce(cls, item: DagNodeLike[T]) -> typing.Self:
        key, deps = item
        return cls(key=key, deps=list(deps))


def _as_dag_nodes[T](
    obj: cabc.Iterable[DagNodeLike[T]] | dict[T, cabc.Iterable[T]],
) -> cabc.Generator[DagNode[T]]:
    if isinstance(obj, cabc.Mapping):
        obj = typing.cast(cabc.Iterable[DagNodeLike[T]], obj.items())

    for key, deps in obj:
        yield DagNode(key=key, deps=list(deps))


def topo_sort[T](
    graph: cabc.Iterable[DagNodeLike[T]] | dict[T, cabc.Iterable[T]], /
) -> list[T]:
    """
    Create a topological sorted list from the given `graph`.
    Uses `TopologicalSorter` under the hood.
    Used this instead of `TopologicalSorter` when data is not `Hashable`.
    """

    node_list = list(_as_dag_nodes(graph))
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
    graph: cabc.Sequence[DagNode[T]],
) -> dict[int, DagNode[T]]:
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
