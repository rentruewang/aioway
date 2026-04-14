# Copyright (c) AIoWay Authors - All Rights Reserved

import abc
import typing
from collections import abc as cabc

import numpy as np
import tensordict as td
import torch

from aioway._common import dcls_no_eq_no_repr, is_list_of
from aioway.ctx import to_fake_tensordict
from aioway.schemas import attr_set

from .fn import DoFn
from .tensors import TensorFn

__all__ = ["TensorDictFn", "tdict"]


class TensorDictFn(DoFn[td.TensorDict], cabc.Mapping[str, TensorFn], abc.ABC):
    @typing.overload
    def __getitem__(self, key: str) -> TensorFn: ...

    @typing.overload
    def __getitem__(self, key: list[str]) -> TensorDictFn: ...

    @typing.no_type_check
    def __getitem__(self, key):
        if isinstance(key, str):
            self.__check_keys(key)

            def get_col(tdict: td.TensorDict) -> torch.Tensor:
                return tdict[key]

            return LambdaTensorFn(self, get_col)

        if (
            False
            or isinstance(key, slice | np.ndarray)
            or is_list_of(int)(key)
            or (isinstance(key, torch.Tensor) and key.dtype != torch.bool)
        ):

            def get_rows(tdict: td.TensorDict) -> td.TensorDict:
                return tdict[key]

            return LambdaTensorDictFn(self, get_rows)

        if isinstance(key, torch.Tensor) and key.dtype == torch.bool:
            return BooleanIndexTensorDictFn(self, key)

        if isinstance(key, TensorFn):
            return GatherTensorDictFn(self, key)

        if is_list_of(str)(key):
            return self.select(*key)

        raise TypeError(f"Does not handle {key=}, {type(key)=}.")

    @typing.override
    def __len__(self) -> int:
        return len(self.attrs)

    @typing.override
    def __contains__(self, key: object, /) -> bool:
        if isinstance(key, str):
            return key in self.keys()

        return False

    @typing.override
    def __iter__(self):
        yield from self.keys()

    @typing.override
    def keys(self):
        return self.attrs.keys()

    @property
    def attrs(self):
        return attr_set(self.preview())

    def rename(self, **renames: str):
        def rename(tdict: td.TensorDict):
            return td.TensorDict(
                {renames.get(key, key): value for key, value in tdict.items()}
            )

        return LambdaTensorDictFn(self, rename)

    def select(self, *keys: str):
        self.__check_keys(*keys)

        def select(tdict: td.TensorDict) -> td.TensorDict:
            return tdict.select(*keys)

        return LambdaTensorDictFn(self, select)

    def zip(self, other: TensorDictFn, /):
        return MergeTensorDictFn(self, other)

    def __check_keys(self, *keys: str):
        for key in keys:
            if key not in self.keys():
                raise KeyError(key)

    @property
    def shape(self):
        return self.attrs.shape_list

    @typing.override
    def _name(self) -> str:
        def components():
            yield self.__class__.__name__
            yield "{"
            for key, val in self.attrs.items():
                yield f"{key}:{val}"
            yield "}"

        return "".join(components())

    @classmethod
    def from_tensordict(cls, data: td.TensorDict) -> TensorDictFn:
        return TensorDictDataFn(data)


def tdict(item: TensorDictFn | td.TensorDict | cabc.Mapping) -> TensorDictFn:
    if isinstance(item, TensorDictFn):
        return item

    if isinstance(item, td.TensorDict):
        return TensorDictFn.from_tensordict(item)

    if isinstance(item, cabc.Mapping):
        tdict = td.TensorDict.from_any(item)
        return TensorDictFn.from_tensordict(tdict)

    raise TypeError(f"Do not know how to handle {type(item)=}.")


class TensorDictDataFn(TensorDictFn):
    "The `Fn` representing a plain `td.TensorDict`."

    def __init__(self, data: td.TensorDict) -> None:
        self.data = data
        self.data.auto_batch_size_()
        assert self.data.ndim, self.data.shape

        super().__init__()

    @typing.override
    def preview(self) -> td.TensorDict:
        return to_fake_tensordict(self.data)

    @typing.override
    def forward(self) -> td.TensorDict:
        return self.data

    @typing.override
    @typing.no_type_check
    def _params_self(self) -> cabc.Generator[torch.Tensor]:
        yield from self.data.values()


@dcls_no_eq_no_repr
class LambdaTensorDictFn(TensorDictFn):
    "The `Fn` representing arbitrary computation on `td.TensorDict`."

    source: TensorDictFn
    function: cabc.Callable[[td.TensorDict], td.TensorDict]

    def __post_init__(self) -> None:
        super().__init__()

    @typing.override
    def forward(self) -> td.TensorDict:
        source = self.source.do()
        return self.function(source)


@dcls_no_eq_no_repr
class LambdaTensorFn(TensorFn):
    "The `Fn` representing arbitrary computation on `td.TensorDict`."

    source: TensorDictFn
    function: cabc.Callable[[td.TensorDict], torch.Tensor]

    def __post_init__(self) -> None:
        super().__init__()

    @typing.override
    def forward(self) -> torch.Tensor:
        source = self.source.do()
        return self.function(source)


@dcls_no_eq_no_repr
class GatherTensorDictFn(TensorDictFn):

    source: TensorDictFn
    index: TensorFn | torch.Tensor

    @typing.override
    def forward(self) -> td.TensorDict:
        from .de import eager

        source = eager(self.source)
        index = eager(self.index)
        return source[index]


@dcls_no_eq_no_repr
class BooleanIndexTensorDictFn(GatherTensorDictFn):

    @typing.override
    def preview(self) -> td.TensorDict:
        return self.source.preview()


@dcls_no_eq_no_repr
class MergeTensorDictFn(TensorDictFn):

    left: TensorDictFn
    right: TensorDictFn

    @typing.override
    def forward(self):
        left = self.left.do()
        right = self.right.do()
        return td.merge_tensordicts(left, right)
