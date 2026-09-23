# Copyright (c) AIoWay Authors - All Rights Reserved

import contextlib as ctxl
import typing
from collections import abc as cabc

import tensordict as td
import torch

from aioway.torch._utils import tcol_to_tdict
from aioway.torch.overrides import is_real

from .attrs import Attr, AttrDict, parse_attr

__all__ = ["fake_cache_schema_attr"]


@ctxl.contextmanager
def fake_cache_schema_attr() -> cabc.Generator[_FakeAttrSchemaCache]:
    """
    In the context, activate fake tensor to `Attr` converter cache.
    """

    cache = _FakeAttrSchemaCache()

    try:
        yield cache

    # After done, destroy the cache to free up fake tensor explicitly.
    finally:
        cache.__init__()


@typing.final
class _FakeAttrSchemaCache:
    """
    A cache that converts fake tensors to `Attr`.
    """

    def __init__(self) -> None:
        """
        The fake tensor cache.
        """

        self._cache: dict[int, Attr] = {}

        # Storing tensors to prevent reuse of `id` due to free.
        self._tensors: dict[int, torch.Tensor] = {}

    def __contains__(self, item: object) -> bool:
        return id(item) in self._tensors

    def __len__(self) -> int:
        return len(self._cache)

    @typing.overload
    def __call__(self, item: torch.Tensor) -> Attr: ...

    @typing.overload
    def __call__(self, item: td.TensorDictBase) -> AttrDict: ...

    @typing.overload
    def __call__(self, item: typing.Any) -> typing.Any: ...

    def __call__(self, item):
        if isinstance(item, torch.Tensor):
            return self.attr(item)

        if td.is_tensor_collection(item):
            return self.schema(item)

        raise TypeError(f"Unhandled {type(item)=}.")

    def attr(self, tensor: torch.Tensor, /) -> Attr:
        "Convert fake tensor to `Attr`. If tensor is real, raise `RuntimeError`."

        if is_real(tensor):
            raise RuntimeError("Only handles fake tensors!")

        if (tensor_id := id(tensor)) not in self._cache:
            self._cache[tensor_id] = parse_attr(tensor)
            self._tensors[tensor_id] = tensor

        return self._cache[tensor_id]

    def schema(self, tcol) -> AttrDict:
        """
        Convert a fake tensor collection to a `Schema`
        """

        if not td.is_tensor_collection(tcol):
            raise TypeError("Only accepts tensor collection!")

        if is_real(tcol):
            raise RuntimeError("Only handles fake tensor collection!")

        tdict = tcol_to_tdict(tcol)

        result = {}

        for key, tensor in tdict.items():
            result[key] = self(tensor)

        return AttrDict(result)
