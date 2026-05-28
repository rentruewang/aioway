# Copyright (c) AIoWay Authors - All Rights Reserved

"The [h]igh level [o]peration [p]review class."

import abc
import contextlib as ctxl
import dataclasses as dcls
import typing
from collections import abc as cabc

from aioway._fn import thunk_dcls
from aioway._utils import AnyDict, Dag, DagNode, dcls_asdict, decomp_flatten, topo_sort

__all__ = ["Hop", "HopDag", "hop_cache_on", "hop_cache"]

_hop_cache: AnyDict[Hop] | None = None
"The cache instance for `Hop`."


@thunk_dcls
class Hop(abc.ABC):
    """
    `Hop` is the node that would be evaluated during run time.
    It will output `torch.Tensor` a container that makes up of them.
    """

    def __hash__(self) -> int:
        return id(self)

    @typing.final
    def __call__(self) -> object:
        if _hop_cache is None:
            return self.forward()

        # Do the caching if enabled.

        cache = hop_cache()

        if self not in cache:
            cache[self] = self.forward()

        return cache[self]

    @abc.abstractmethod
    def forward(self) -> object:
        raise NotImplementedError


@ctxl.contextmanager
def hop_cache_on():
    """
    Turn on caching for `Hop`. Everytime you call `hop_cache_on`,
    a new scope is created and so a new cache is created.
    (The old cache still stays in memory so it'll still be "active").
    """

    global _hop_cache

    _hop_cache = AnyDict[Hop](Hop)
    try:
        yield _hop_cache
    finally:
        _hop_cache = None


def hop_cache() -> AnyDict[Hop]:
    """
    The active cache for `Hop`. If there is no active session, raise `RuntimeError`.
    """

    if _hop_cache is None:
        raise RuntimeError("`hop_cache` can only be called in `hop_cache_on` scope.")

    return _hop_cache


@dcls.dataclass
class HopDag:
    """
    The DAG of `HopInit`s or `HopFwd`s.
    """

    dag: Dag[Hop]
    "The ordered nodes."

    def __len__(self):
        return len(self.dag)

    def __iter__(self) -> cabc.Generator[Hop]:
        yield from self.dag

    def __call__(self) -> list[object]:
        "Evaluating the `HopInit`/`HopFwd`."

        items = self.dag.items
        return [node() for node in items]

    @property
    def input_nodes(self) -> list[Hop]:
        "Get the input nodes."

        return self.dag.inputs_items

    @property
    def output_nodes(self) -> list[Hop]:
        "Get the output nodes."

        return self.dag.outputs_items

    @classmethod
    def from_list_of_nodes(cls, nodes: list[Hop]) -> typing.Self:
        dag_nodes = [_to_dag_node(node) for node in nodes]
        dag: Dag[Hop] = topo_sort(dag_nodes)
        return cls(dag)


def _to_dag_node(hop: Hop) -> DagNode[Hop]:
    """
    The dependent `Hop`s. It's decomposed from the dataclass members.
    """

    # Flatten the dict version s.t. we do not include `self`.
    deps: list[Hop] = list(decomp_flatten(dcls_asdict(hop), Hop))
    return DagNode(key=hop, deps=deps)
