# Copyright (c) AIoWay Authors - All Rights Reserved

"The [h]igh level [o]peration [p]review class."

import abc
import dataclasses as dcls
import typing

from aioway._fn import thunk_dcls
from aioway._utils import Dag, DagNode, decomp_flatten, topo_sort

__all__ = ["HopInit", "HopFwd"]


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
        return self.init()

    @abc.abstractmethod
    def init(self) -> HopFwd:
        raise NotImplementedError

    @typing.override
    def to_dag_node(self) -> DagNode[typing.Self]:
        deps = list(decomp_flatten(self, HopInit))
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
        return self.forward()

    @abc.abstractmethod
    def forward(self) -> object:
        raise NotImplementedError

    @typing.override
    def to_dag_node(self) -> DagNode[typing.Self]:
        """
        The dependent `Hop`s. It's decomposed from the dataclass members.
        """

        deps = list(decomp_flatten(self, HopFwd))
        return DagNode(self, deps)


@dcls.dataclass
class HopDag[T: (HopInit, HopFwd)]:
    """
    The DAG of `HopInit`s or `HopFwd`s.
    """

    dag: Dag[T]
    "The ordered nodes."

    def __call__(self):
        "Evaluating the `HopInit`/`HopFwd`."

        items = self.dag.items
        return [node() for node in items]

    @classmethod
    def from_list_of_nodes(cls, nodes: list[T]) -> typing.Self:
        dag_nodes = [node.to_dag_node() for node in nodes]
        dag = topo_sort(dag_nodes)
        return cls(dag)
