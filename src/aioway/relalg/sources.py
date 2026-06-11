# Copyright (c) AIoWay Authors - All Rights Reserved

"The `Stream` that records past histories and supports random access."

import abc
import functools
import logging
import typing
from collections import abc as cabc

import tensordict as td

from aioway.attrs import AttrDict
from aioway.hop import TdictHop, hop_dcls

__all__ = ["BoundedHop", "SourceListHop"]

LOGGER = logging.getLogger(__name__)


@hop_dcls
class BoundedHop(TdictHop, abc.ABC):
    """
    A stream with `__len__` and `__getitem__`.
    """

    @abc.abstractmethod
    def __len__(self) -> int:
        "The number of batches saved in the current `Stream`."

        raise NotImplementedError

    @abc.abstractmethod
    def __getitem__(self, key: int, /) -> td.TensorDict:
        """
        Get individual items. Does not support slice input.

        Args:
            idx: An integer. Must be in the range `[-len(self), len(self))`.

        Returns:
            The `td.TensorDict` batch.
        """

        raise NotImplementedError


@hop_dcls
class SourceListHop(BoundedHop):
    "A `Stream` backed by a list of `TensorDict`."

    sequence: cabc.Sequence[td.TensorDict]
    "List of `td.TensorDict`s."

    @typing.override
    def __len__(self) -> int:
        return self.size

    @typing.override
    def __getitem__(self, idx: int) -> td.TensorDict:
        return self.sequence[idx]

    @property
    @typing.override
    def size(self) -> int:
        return len(self.sequence)

    @property
    @typing.override
    def attrs(self) -> AttrDict:
        return self._schema

    @functools.cached_property
    def _schema(self) -> AttrDict:
        schemas = [AttrDict.parse(chunk) for chunk in self.sequence]

        if len({*schemas}) == 1:
            return schemas[0]

        raise ValueError("Chunks should have the same schema.")

    def iterate(self):
        for batch in self.sequence:
            yield batch

    @classmethod
    def exhaust(cls, stream: TdictHop) -> typing.Self:
        return cls(list(stream))
