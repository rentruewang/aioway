# Copyright (c) AIoWay Authors - All Rights Reserved

"The [h]igh level [o]peration [p]review class."

import abc
import contextlib as ctxl
import dataclasses as dcls
import typing
from collections import abc as cabc

from aioway._fn import thunk_dcls
from aioway._utils import Dag, DagNode, dcls_asdict, decomp_flatten, topo_sort

__all__ = [
    "HopInit",
    "HopFwd",
    "HopDag",
    "HopDagNode",
    "hop_init_dcls",
    "cache_hop_init_output",
    "cache_hop_fwd_output",
    "hop_init_cache",
    "hop_fwd_cache",
]

_hop_init_cache: dict[int, HopFwd] | None = None
"The cache instance for `HopInit`."

_hop_fwd_cache: dict[int, object] | None = None
"The cache instance for `HopFwd`."


class HopDagNode[T](abc.ABC):
    "Defines the `to_dag_node` functionality."

    @abc.abstractmethod
    def to_dag_node(self: T) -> DagNode[T]:
        raise NotImplementedError


@typing.dataclass_transform(kw_only_default=True)
def hop_init_dcls(cls):
    return dcls.dataclass(match_args=True, kw_only=True)(cls)


@hop_init_dcls
class HopInit(HopDagNode["HopInit"], abc.ABC):
    """
    `HopInit` stands for [h]igh level [o]peration [p]review node.
    It is essentailly an unevaluated expression that supports inspection.
    """

    def __hash__(self) -> int:
        return id(self)

    @typing.final
    def __call__(self) -> HopFwd:
        return _maybe_save_cache(self.init, id(self), _hop_init_cache)

    @abc.abstractmethod
    def init(self) -> HopFwd:
        raise NotImplementedError

    @typing.override
    def to_dag_node(self) -> DagNode[typing.Self]:
        # Flatten the dict version s.t. we do not include `self`.
        deps = list(decomp_flatten(dcls_asdict(self), HopInit))
        return DagNode(self, deps)


@thunk_dcls
class HopFwd(HopDagNode["HopFwd"], abc.ABC):
    """
    `HopFwd` is the node that would be evaluated during run time.
    It is initialized by `HopInit`.
    """

    def __hash__(self) -> int:
        return id(self)

    @typing.final
    def __call__(self) -> object:
        return _maybe_save_cache(self.forward, id(self), _hop_fwd_cache)

    @abc.abstractmethod
    def forward(self) -> object:
        raise NotImplementedError

    @typing.override
    def to_dag_node(self) -> DagNode[typing.Self]:
        """
        The dependent `Hop`s. It's decomposed from the dataclass members.
        """

        # Flatten the dict version s.t. we do not include `self`.
        deps = list(decomp_flatten(dcls_asdict(self), HopFwd))
        return DagNode(self, deps)


@ctxl.contextmanager
def cache_hop_init_output():
    "Turn on caching for `HopInit`."
    global _hop_init_cache
    _hop_init_cache = {}
    try:
        yield _hop_init_cache
    finally:
        _hop_init_cache = None


def hop_init_cache():
    "The active cache for `HopInit`. If `None` caching is off."
    return _hop_init_cache


@ctxl.contextmanager
def cache_hop_fwd_output():
    "Turn on caching for `HopFwd`."
    global _hop_fwd_cache
    _hop_fwd_cache = {}
    try:
        yield _hop_fwd_cache
    finally:
        _hop_fwd_cache = None


def hop_fwd_cache():
    "The active cache for `HopFwd`. If `None` caching is off."
    return _hop_fwd_cache


def _maybe_save_cache[T](
    func: cabc.Callable[[], typing.Any],
    hash_key: int,
    cache: dict[int, T] | None = None,
) -> T:
    if cache is None:
        return func()

    if hash_key in cache:
        return cache[hash_key]

    result = func()
    cache[hash_key] = result
    return result


@dcls.dataclass
class HopDag[T: (HopInit, HopFwd)]:
    """
    The DAG of `HopInit`s or `HopFwd`s.
    """

    dag: Dag[T]
    "The ordered nodes."

    def __len__(self):
        return len(self.dag)

    def __iter__(self):
        yield from self.dag

    def __call__(self):
        "Evaluating the `HopInit`/`HopFwd`."

        items = self.dag.items
        return [node() for node in items]

    @property
    def input_nodes(self) -> list[T]:
        "Get the input nodes."

        return self.dag.inputs_items

    @property
    def output_nodes(self) -> list[T]:
        "Get the output nodes."

        return self.dag.outputs_items

    @classmethod
    def from_list_of_nodes(cls, nodes: list[T]) -> typing.Self:
        dag_nodes = [node.to_dag_node() for node in nodes]
        dag: Dag[T] = topo_sort(dag_nodes)
        return cls(dag)
