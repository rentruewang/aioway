# Copyright (c) AIoWay Authors - All Rights Reserved

"The index (currently only faiss)."

import abc
import typing
from collections import abc as cabc

import numpy as np
import torch

from aioway._utils import FloatArray, IntArray
from aioway.hop import TensorHop, hop_dcls

__all__ = ["Index", "FaissIndex", "FaissIndexHop"]


class Index(abc.ABC):
    def __init__(self, dim: int):
        self._dim = dim

    @abc.abstractmethod
    def train(self, x: FloatArray) -> None:
        raise NotImplementedError

    def search(self, query: FloatArray, k: int = 1) -> tuple[FloatArray, IntArray]:
        if not isinstance(query, np.ndarray):
            raise TypeError(f"{type(query)=} should be numpy.")

        if k <= 0:
            raise ValueError(f"The neighbor count {k} must be positive.")

        assert query.ndim == 2
        assert query.shape[-1] == self._dim

        dist, idx = self._search(query=query, k=k)
        assert dist.ndim == 2, dist.shape
        assert dist.shape[1] == k, dist.shape
        assert idx.ndim == 2, idx.shape
        assert idx.shape[1] == k, idx.shape
        return dist, idx

    @abc.abstractmethod
    def _search(self, query: FloatArray, k: int = 1) -> tuple[FloatArray, IntArray]:
        raise NotImplementedError

    @classmethod
    def from_tensors(cls, training: torch.Tensor) -> typing.Self:
        assert training.ndim == 2
        dim = training.shape[1]
        inst = cls(dim)
        inst.train(training.cpu().numpy().astype("float32"))
        return inst

    @classmethod
    def from_hop(cls, tensors: TensorHop) -> typing.Self:
        training = torch.cat(list(tensors))
        return cls.from_tensors(training)


class FaissIndex(Index):
    """
    An index exposing `hnswlib` API.
    """

    def __init__(self, dim: int, spec: str = "Flat"):
        import faiss

        super().__init__(dim)
        self._idx = faiss.index_factory(dim, spec)

    def train(self, x: FloatArray):
        self._idx.train(x)

    def _search(self, query: FloatArray, k: int = 1) -> tuple[FloatArray, IntArray]:
        return self._idx.search(query, k)


@hop_dcls
class FaissIndexHop(TensorHop):
    index: FaissIndex
    """
    The index backed by the `faiss` library.
    """

    source: torch.Tensor
    """
    The source that would be queried.
    """

    query: TensorHop
    """
    The querying tensor.
    """

    k: int
    """
    The number of neighbors to return.
    """

    def iterate(self) -> cabc.Generator[torch.Tensor]:
        for query in self.query:
            _, index = self.index.search(query.cpu().numpy(), k=self.k)
            yield self.source[index]
