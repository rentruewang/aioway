# Copyright (c) AIoWay Authors - All Rights Reserved

"Chunk is a heterogenious collection of in-memory tensor batches."

import dataclasses as dcls
import logging
import typing
from collections import abc as cabc

import tensordict as td
import torch

from aioway._common import is_dict_of_str_to
from aioway._common.typing.checks import is_list_of
from aioway.schemas import AttrSetLike, attr_set

from .vectors import Vector

__all__ = ["Chunk"]

LOGGER = logging.getLogger(__name__)


type TensorDictLike = td.TensorDict | dict[str, torch.Tensor]
type ChunkLike = Chunk | dict[str, Vector]


@typing.final
@dcls.dataclass(frozen=True)
class Chunk(cabc.Mapping[str, Vector]):
    """
    A `Chunk` represents a batch of data, following a specific scheam.

    For now, it is immutable as implementing mutable interface is harder.
    This can change in the future.
    """

    data: td.TensorDict
    "The underlying data."

    @typing.override
    def __len__(self) -> int:
        "The length of the current batch."

        return len(self.data)

    @typing.override
    def __repr__(self) -> str:
        return f"{self.attrs!r}({len(self)})"

    @typing.override
    def __eq__(self, rhs: object) -> bool:
        # Check if schema and data are both equal.
        if isinstance(rhs, Chunk):
            return self.attrs == rhs.attrs and (self.data == rhs.data).all()

        # If tensordict, don't compare schema.
        if isinstance(rhs, dict | td.TensorDict):
            try:
                return (self.data == rhs).all()
            except KeyError:
                return NotImplemented

        return NotImplemented

    def __getitem__(self, key):
        if isinstance(key, Vector):
            key = key.data

        if isinstance(key, str):
            result = self.data[key]
            assert isinstance(result, torch.Tensor)
            return Vector(result)
        elif is_list_of(str)(key):
            result = self.data.select(*key)
            return type(self)(result)
        else:
            return type(self)(self.data[key])

    @typing.override
    def __iter__(self) -> cabc.Iterator[str]:
        return iter(self.attrs)

    def select(self, *names: str):
        return Chunk(self.select(*names))

    def column(self, col: str):
        return Vector(self[col].forward())

    def rename(self, **renames: str):
        if not renames:
            return self

        def rename(tdict: td.TensorDict):
            return td.TensorDict(
                {renames.get(key, key): value for key, value in tdict.items()}
            )

        return Chunk(rename(self.data))

    def zip(self, rhs: typing.Self):
        return Chunk(td.merge_tensordicts(self.data, rhs.data))

    @property
    def shape(self) -> torch.Size:
        return self.data.shape

    def torch(self):
        return self.data

    @classmethod
    def cat(cls, chunks: cabc.Sequence[typing.Self]) -> typing.Self:
        if not chunks:
            raise ValueError("Given an empty sequence. Not sure what to do.")

        if len({tuple(chunk.attrs.device_list) for chunk in chunks}) != 1:
            raise ValueError("Chunks should have the same devices before joining.")

        if len({tuple(chunk.attrs.dtype_list) for chunk in chunks}) != 1:
            raise ValueError("Chunks should have the same dtypes before joining.")

        schema = chunks[0].attrs

        data = td.cat([c.data for c in chunks])

        return cls.from_data_schema(schema=schema, data=data)

    @property
    def attrs(self):
        return attr_set(self.data)

    @classmethod
    def from_data_schema(cls, data: TensorDictLike, schema: AttrSetLike) -> typing.Self:
        td = _as_tensordict(data)
        td.auto_batch_size_()
        td.auto_device_()
        return cls(data=td)

    @classmethod
    def from_mapping(cls, chunk: ChunkLike) -> typing.Self:
        if isinstance(chunk, cls):
            return chunk

        if _is_mapping_of_vector(chunk):
            data = {key: chunk[key].torch() for key in chunk.keys()}
            schema = {key: chunk[key].typeof() for key in chunk.keys()}
            return cls.from_data_schema(data=data, schema=schema)

        raise TypeError(type(chunk))


def _as_tensordict(data: TensorDictLike, /) -> td.TensorDict:
    if isinstance(data, td.TensorDict):
        return data

    _is_dict_of_tensor = is_dict_of_str_to(torch.Tensor)
    if _is_dict_of_tensor(data):
        return td.TensorDict(data)

    raise TypeError(type(data))


@typing.no_type_check
def _is_mapping_of_vector(obj) -> typing.TypeGuard[dict[str, Vector]]:
    # Wrapper function because `mypy` doesn't do well with abstract type guards.
    return is_dict_of_str_to(Vector)(obj)
