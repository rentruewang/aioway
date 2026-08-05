# Copyright (c) AIoWay Authors - All Rights Reserved

"`Space`s for `td.TensorClass`."

import abc
import dataclasses as dcls
import typing

import tensordict as td
import torch

from aioway._api import public_api

from .spaces import Space, space_dcls
from .tdicts import IsTdictSpace
from .tensors import IsTensorSpace

__all__ = ["space_for_tcls"]


@public_api
def space_for_tcls(cls: type) -> _TClsSpace:
    """
    Get the space for a tensorclass.
    """

    if not td.is_tensorclass(cls):
        raise TypeError(f"{cls=} is not a tensorclass.")

    assert dcls.is_dataclass(cls), "A tensorclass is already a dataclass."

    hints = typing.get_type_hints(cls)
    members: dict[str, Space] = {}

    for name, hint in hints.items():
        assert isinstance(hint, type)
        members[name] = _space_for_type(hint)

    return _TClsSpace(members)


def _space_for_type(cls: type) -> Space:
    if cls is torch.Tensor:
        return IsTensorSpace()

    if cls is td.TensorDict:
        return IsTdictSpace()

    if td.is_tensorclass(cls):
        return space_for_tcls(cls)

    raise ValueError(f"Unhandled cls type: {cls=}")


@space_dcls
class _TClsSpace[T: td.TensorClass](Space[T], abc.ABC):
    "A `Space` that checks a `td.TensorClass`."

    members: dict[str, Space]
    "The members of the `td.TensorClass`."

    @typing.final
    def contains(self, inst: T) -> bool:
        if not td.is_tensorclass(inst):
            return False

        assert dcls.is_dataclass(inst)
        fields = dcls.fields(inst)

        if self.members.keys() != {f.name for f in fields}:
            return False

        for key, space in self.members.items():
            if getattr(inst, key) not in space:
                return False

        return True

    def _sample_n(self, n: int):
        # This is not exactly what we are supposed to return,
        # but will suffice for now.
        return dict(self._sample_n_iter(n))

    def _sample_n_iter(self, n: int):
        for key, val in self.members.items():
            yield key, val.sample(n)
