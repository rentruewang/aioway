# Copyright (c) AIoWay Authors - All Rights Reserved

import contextlib as ctxl
import dataclasses as dcls
from collections import abc as cabc

import torch
from IPython.core.guarded_eval import typing
from torch import nn

from aioway._utils import register_module_forward_hook
from aioway._utils.types import AnyDict
from aioway.tensors import fake_mode
from aioway.tensors.routes import route_aten_thunk

__all__ = ["capture_module_hist", "ModuleInOutThunk", "ModuleInOutHist"]


@dcls.dataclass(frozen=True)
class ModuleInOutThunk:
    module: nn.Module

    input: tuple[torch.Tensor, ...]
    output: torch.Tensor

    def inputs(self) -> tuple[torch.Tensor, ...]:
        return self.input

    def __call__(self) -> torch.Tensor:
        return self.output


@dcls.dataclass(frozen=True)
class ModuleInOutHist:
    """
    The history to track input and output of a `nn.Module` during running.

    Will need to integrate with `Hist` in the future.
    """

    history: list[ModuleInOutThunk] = dcls.field(default_factory=list)
    """
    The history encountered.
    """

    output_index: AnyDict[torch.Tensor, int] = dcls.field(default_factory=AnyDict)
    """
    The index of each tensor where `thunk.output = tensor`.

    Since this graph is about expression, we only care about the output.
    """

    def __len__(self) -> int:
        return len(self.history)

    def __getitem__(self, idx: int) -> ModuleInOutThunk:
        return self.history[idx]

    def append(self, thunk: ModuleInOutThunk) -> None:
        length = len(self)
        self.history.append(thunk)
        self.output_index[thunk.output] = length

    def thunk_of(self, output: torch.Tensor) -> ModuleInOutThunk:
        idx = self.output_index[output]
        return self[idx]

    def module_forward_hook(
        self, module: nn.Module, input: tuple[torch.Tensor, ...], output: torch.Tensor
    ) -> None:
        thunk = ModuleInOutThunk(module=module, input=input, output=output)
        self.append(thunk)

    @ctxl.contextmanager
    def register_module_forward_hook(self) -> cabc.Generator[typing.Self]:
        with register_module_forward_hook(self.module_forward_hook):
            yield self


@ctxl.contextmanager
def capture_module_hist():
    hist = ModuleInOutHist()

    with fake_mode(), route_aten_thunk.activate(), hist.register_module_forward_hook():
        yield hist
