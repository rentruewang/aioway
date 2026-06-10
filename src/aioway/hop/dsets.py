# Copyright (c) AIoWay Authors - All Rights Reserved

"The dataset related `Hop`s."

import functools
import typing
from collections import abc as cabc

import torch

from aioway.attrs import Attr

from .hop import TensorHop, hop_dcls

__all__ = ["TensorListHop"]


@hop_dcls
class TensorListHop(TensorHop):
    "A `Hop` backed by a list of `torch.Tensor`."

    sequence: cabc.Sequence[torch.Tensor]
    "List of `torch.Tensor`s."

    @property
    @typing.override
    def size(self) -> int:
        return len(self.sequence)

    @property
    @typing.override
    def attr(self) -> Attr:
        return self._schema

    @functools.cached_property
    def _schema(self) -> Attr:
        schemas = [Attr.parse(tensor) for tensor in self.sequence]

        if len({*schemas}) == 1:
            return schemas[0]

        raise ValueError("Chunks should have the same schema.")

    def iterate(self):
        for batch in self.sequence:
            yield batch
