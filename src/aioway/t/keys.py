# Copyright (c) AIoWay Authors - All Rights Reserved

import dataclasses as dcls
import typing

import torch

from aioway.t.attrs import Attr, parse_attr

__all__ = ["TKey", "THash"]


@dcls.dataclass(frozen=True)
class TKey:
    "Wraps a tensor as a key, hashing and comparing its content."

    tensor: torch.Tensor

    def __hash__(self) -> int:
        "Hash of the tensor's attribute and content."

        return hash((self.attr, THash()(self.tensor)))

    def __eq__(self, other) -> bool:
        "Equal if the attributes and contents are equal."

        if isinstance(other, torch.Tensor):
            other = TKey(other)

        if not isinstance(other, TKey):
            return NotImplemented

        if self.tensor is other.tensor:
            return True

        if self.attr != other.attr:
            return False

        return torch.equal(self.tensor, other.tensor)

    @property
    def attr(self) -> Attr:
        "The attribute of the tensor."

        return parse_attr(self.tensor)


# Hashing. ====

_INTS = {1: torch.int8, 2: torch.int16, 4: torch.int32, 8: torch.int64}


@dcls.dataclass(frozen=True)
class THash:
    """
    Hashes the content of a tensor, on the tensor's device.

    Source - https://stackoverflow.com/a/77213071
    Posted by Mateen Ulhaq, modified by community. See post 'Timeline' for change history
    Retrieved 2026-11-27, License - CC BY-SA 4.0
    """

    mult: int = 6364136223846793005
    inc: int = 1

    @torch.no_grad()
    def __call__(self, tensor: torch.Tensor) -> int:
        "Hash the content of `tensor`, 0 if it's empty."

        if not tensor.numel():
            return 0

        x = self._as_int64(tensor)
        while x.ndim > 0:
            x = self._reduce(x)

        item: typing.Any = x.item()
        return item

    def _as_int64(self, x: torch.Tensor) -> torch.Tensor:
        "Reinterpret the bits of `x` as integers, then widen to `int64`."

        x = x.detach().contiguous()
        x = x.view(_INTS[min(x.element_size(), 8)])
        return x.to(torch.int64)

    def _reduce(self, x: torch.Tensor) -> torch.Tensor:
        "Reduce the last axis."

        assert x.dtype == torch.int64
        acc = torch.zeros_like(x[..., 0])
        for i in range(x.shape[-1]):
            acc *= self.mult
            acc += self.inc
            acc += x[..., i]
        return acc
