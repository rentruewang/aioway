# Copyright (c) AIoWay Authors - All Rights Reserved

import torch

import dataclasses as dcls
from aioway.t.attrs import Attr, parse_attr


@dcls.dataclass(frozen=True)
class TKey:
    tensor: torch.Tensor

    def __hash__(self) -> int:
        raise NotImplementedError

    def __eq__(self, other) -> bool:
        if isinstance(other, torch.Tensor):
            other = TKey(other)

        if not isinstance(other, TKey):
            return NotImplemented

        if self.tensor is other.tensor:
            return True

        if self.attr != other.attr:
            return False

        # Use this:
        # Source - https://stackoverflow.com/a/77213071
        # Posted by Mateen Ulhaq, modified by community. See post 'Timeline' for change history
        # Retrieved 2026-11-27, License - CC BY-SA 4.0

        # import torch
        # from torch import Tensor

        # MULTIPLIER = 6364136223846793005
        # INCREMENT = 1
        # MODULUS = 2**64

        # def hash_tensor(x: Tensor) -> Tensor:
        #     assert x.dtype == torch.int64
        #     while x.ndim > 0:
        #         x = _reduce_last_axis(x)
        #     return x

        # @torch.no_grad()
        # def _reduce_last_axis(x: Tensor) -> Tensor:
        #     assert x.dtype == torch.int64
        #     acc = torch.zeros_like(x[..., 0])
        #     for i in range(x.shape[-1]):
        #         acc *= MULTIPLIER
        #         acc += INCREMENT
        #         acc += x[..., i]
        #         # acc %= MODULUS  # Not really necessary.
        #     return acc

        raise NotImplementedError

    @property
    def attr(self) -> Attr:
        return parse_attr(self.tensor)
