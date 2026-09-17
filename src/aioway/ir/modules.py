# Copyright (c) AIoWay Authors - All Rights Reserved

import contextlib as ctxl
import dataclasses as dcls
import typing
from collections import abc as cabc

import torch
from torch import nn

from aioway._utils import AnyDict, register_module_forward_hook
from aioway.tensors import (
    clone_in_fake_mode,
    fake_mode,
    find_nested_tensors,
    route_aten_thunk,
)

__all__ = ["capture_module_hist", "ModuleInOutThunk", "ModuleInOutHist"]


@dcls.dataclass(frozen=True)
class ModuleInOutThunk:
    module: nn.Module

    input: typing.Any
    output: typing.Any

    def inputs(self) -> cabc.Generator[torch.Tensor]:
        yield from find_nested_tensors(self.input)

    def outputs(self) -> cabc.Generator[torch.Tensor]:
        yield from find_nested_tensors(self.output)


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
        outputs = tuple(thunk.outputs())

        # Check if the keys already exists,
        # should be unique due to cloning in fake mode.
        if not self.output_index.keys().isdisjoint(outputs):
            raise KeyError(f"Impossible conflicting output at {self.history[-1]}.")

        length = len(self)
        self.history.append(thunk)

        for output in outputs:
            self.output_index[output] = length

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
def capture_module_hist() -> cabc.Generator[ModuleInOutHist]:
    with ctxl.ExitStack() as stack:
        for mode in [
            fake_mode(),
            clone_in_fake_mode.activate(),
            route_aten_thunk.activate(),
            (hist := ModuleInOutHist()).register_module_forward_hook(),
        ]:
            stack.enter_context(mode)

        yield hist
