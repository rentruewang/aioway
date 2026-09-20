# Copyright (c) AIoWay Authors - All Rights Reserved

import dataclasses as dcls
import typing
from collections import abc as cabc

import torch

from aioway.torch import (
    TorchDispThunk,
    TorchFuncThunk,
    replace_tensors_with_attr,
)

__all__ = []


@dcls.dataclass(frozen=True)
class FakeThunkResult[F: TorchFuncThunk | TorchDispThunk]:
    "Stores the thunk and output."

    thunk: F
    "The `Thunk` that has been called. Only takes in fake tensors."

    result: object
    "The output that `fn` has produced."

    @typing.override
    def __repr__(self) -> str:
        result = replace_tensors_with_attr(self.result)
        return f"{self.thunk!r} -> {result}"

    def inputs(self) -> cabc.Generator[torch.Tensor]:
        yield from self.thunk.inputs()
