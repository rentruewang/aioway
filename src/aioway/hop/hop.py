# Copyright (c) AIoWay Authors - All Rights Reserved

"The [h]igh level [o]peration [p]review class."

from jupyter_lsp.specs import r

import abc
import contextlib as ctxl
import dataclasses as dcls
import typing
from collections import abc as cabc

from aioway._utils import AnyDict, DagNode, dcls_asdict, decomp_flatten, topo_sort

__all__ = ["Hop", "HopDag", "hop_cache_on", "hop_cache"]

_hop_cache: AnyDict[Hop] | None = None
"The cache instance for `Hop`."


@typing.dataclass_transform()
def hop_dcls(cls: type):
    return dcls.dataclass(match_args=False)(cls)


@hop_dcls
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
        """
        The forwarding logic. Should invoke dependencies' `__call__` methods.
        """

        raise NotImplementedError

    def rebuild(self):
        """
        Rebuild the current `Hop`. This is useful when you are switching contexts,
        e.g. switching on real mode after configuring the `HopDag` in fake mode.
        """

        copied = self._rebuild()
        assert copied is not self
        return copied

    @abc.abstractmethod
    def _rebuild(self):

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


@dcls.dataclass(frozen=True)
class HopNode:
    hop: Hop

    @property
    def input_hops(self) -> cabc.Iterator[Hop]:
        yield from decomp_flatten(dcls_asdict(self.hop), Hop)

    @property
    def is_input(self) -> bool:
        return not list(self.input_hops)


@dcls.dataclass
class HopDag:
    """
    The DAG of `HopInit`s or `HopFwd`s.
    """

    dag: list[HopNode]
    "The ordered nodes."

    def __len__(self):
        return len(self.dag)

    def __iter__(self) -> cabc.Generator[Hop]:
        yield from self.dag

    def __call__(self) -> list[object]:
        "Evaluating the `HopInit`/`HopFwd`."

        return [node.hop() for node in self.dag]

    @property
    def input_nodes(self) -> list[Hop]:
        "Get the input nodes."

        return [len(list(node.input_hops())) for node in self.dag]

    @property
    def output_nodes(self) -> list[Hop]:
        "Get the output nodes."

        return self.dag.outputs_items

    @classmethod
    def from_list_of_nodes(cls, nodes: list[Hop]) -> typing.Self:
        dag_nodes = [_to_dag_node(node) for node in nodes]
        dag = topo_sort(dag_nodes)
        return cls([HopNode(hop) for hop in dag])


def _to_dag_node(hop: Hop) -> DagNode[Hop]:
    """
    The dependent `Hop`s. It's decomposed from the dataclass members.
    """

    # Flatten the dict version s.t. we do not include `self`.
    deps: list[Hop] = list(decomp_flatten(dcls_asdict(hop), Hop))
    return DagNode(key=hop, deps=deps)
