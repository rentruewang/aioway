# Copyright (c) AIoWay Authors - All Rights Reserved

"The `Fn`s corresponding to module's."

import abc
import dataclasses as dcls
import typing
from collections import abc as cabc

import torch
from torch import nn
from aioway.previews
from aioway._common import find_nested_tensors
from aioway._common import HasParam

from .fn import Fn, TorchThunk

__all__ = ["NnForwardFn", "NnInitFn", "PreviewFn"]


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
    `NnInitFn` is the "leftover" `nn.Module`s that are not covered by the `Preview` API.
    """

    func: type[nn.Module]
    "The type of `nn.Module`."

    def __post_init__(self):
        super().__post_init__()

        if not isinstance(self.func, type) or not issubclass(self.func, nn.Module):
            raise TypeError(f"{self.func} should be a subclass of `nn.Module`.")


class PreviewFn(HasParam, Fn):
    """
    `PreviewFn` are `Fn` that wrap `Preview`s, which are supported `nn.Module` ops.
    """

    preview: Preview

    @typing.override
    def do(self) -> object:
        raise NotImplementedError

    @typing.override
    def tensors(self) -> cabc.Iterator[torch.Tensor]:
        raise NotImplementedError

    @classmethod
    def find_preview(cls, thunk: NnInitFn) -> typing.Self:
        raise NotImplementedError
