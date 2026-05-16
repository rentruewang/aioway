# Copyright (c) AIoWay Authors - All Rights Reserved

"The `Fn`s corresponding to module's."

import dataclasses as dcls
import typing
from collections import abc as cabc

import torch
from torch import nn

from aioway._common import find_nested_tensors
from aioway.might import Might, find_might

from .fn import TorchThunk

__all__ = ["NnForwardFn", "NnInitFn", "MightFn"]


@dcls.dataclass
class NnForwardFn(TorchThunk[nn.Module]):
    "`NnForwardFn` represents the module calls."

    func: nn.Module
    "The module for the `Fn`."

    @typing.override
    def tensors(self) -> cabc.Iterator[torch.Tensor]:
        yield from find_nested_tensors(self.args)
        yield from find_nested_tensors(self.kwargs)
        yield from self.func.parameters()


class NnInitFn(TorchThunk[type[nn.Module]]):
    """
    `NnInitFn` is the "leftover" `nn.Module`s that are not covered by the `Might` API.
    """

    func: type[nn.Module]
    "The type of `nn.Module`."

    def __post_init__(self):
        super().__post_init__()

        if not isinstance(self.func, type) or not issubclass(self.func, nn.Module):
            raise TypeError(f"{self.func} should be a subclass of `nn.Module`.")


@dcls.dataclass
class MightFn:
    """
    `MightFn` are `Fn` that wrap `Might`s, which are supported `nn.Module` ops.
    """

    might: Might
    "The `Might` instance."

    def do(self) -> object:
        return self.might.do()

    @classmethod
    def find_preview(cls, thunk: NnInitFn) -> typing.Self:
        might = find_might(thunk.func, *thunk.args, **thunk.kwargs)

        if might is NotImplemented:
            return NotImplemented

        else:
            return cls(might)
