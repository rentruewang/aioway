# Copyright (c) AIoWay Authors - All Rights Reserved

"The operator base class."

import abc
import collections
import dataclasses as dcls
import graphlib
import typing
from collections import abc as cabc

import torch

from aioway._utils.decomps import find_nested_tensors
from aioway.fn import Fn

__all__ = []


class Hop(Fn, abc.ABC):
    """
    `Hop` stands for [h]igh level [op]erator, or [h]igh level [o]peration [p]review.
    It is essentailly an unevaluated expression that supports inspection.
    """

    def __hash__(self):
        return id(self)

    @abc.abstractmethod
    def deps(self) -> cabc.Iterator[Hop]:
        raise NotImplementedError

    @abc.abstractmethod
    def do(self) -> object:
        """
        Evaluates the current operator and outputs the results.
        The object must be decomposed into pure tensors (no extra items e.g. primitives).
        """

        raise NotImplementedError


@dcls.dataclass(frozen=True)
class HopDag:
    """
    `HopDag` is a DAG of `Hop`s, ordered in the linear sense.
    """

    hops: cabc.Sequence[Hop]
    "The topologically sorted `Hop`s list."

    @classmethod
    def from_hops(cls, hops: cabc.Iterable[Hop]) -> typing.Self:
        hashes = [hash(hop) for hop in hops]
        hop_to_tensors: dict[int, list[torch.Tensor]] = collections.defaultdict(list)
        for hop in hops:
            hop_to_tensors[hash(hop)].extend(find_nested_tensors(hop))
        tensors = [list(find_nested_tensors(hop)) for hop in hops]
        topo_sorter = graphlib.TopologicalSorter()
        topo_sorter

        raise NotImplementedError
