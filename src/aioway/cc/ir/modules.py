# Copyright (c) AIoWay Authors - All Rights Reserved

from aioway._utils import Stack
import contextlib as ctxl
import dataclasses as dcls
import typing
from collections import abc as cabc

import torch
from torch import nn

from aioway._utils import AnyDict
from aioway.torch import (
    clone_in_fake_mode,
    fake_mode,
    find_nested_tensors,
    register_module_forward_hook,
    register_module_forward_pre_hook,
    route_aten_thunk,
)
from .dags import DoneThunk

__all__ = ["capture_module_hist", "ModuleThunk", "ModuleInOutHist"]


@dcls.dataclass(frozen=True)
class ModuleThunk(DoneThunk):
    """
    Module thunk is a thunk tracking inputs, outputs, and which module calls it.
    """

    _: dcls.KW_ONLY

    parents: tuple[nn.Module, ...]
    "The parent modules that calls this current thunk. It's a stack."


@ctxl.contextmanager
def track_module_thunks():
    yield


class _ModuleInput(typing.NamedTuple):
    module: nn.Module
    input: typing.Any


@dcls.dataclass(frozen=True)
class ModuleTracker:
    stack: Stack[_ModuleInput]
    "The current stack of modules."

    @ctxl.contextmanager
    def __call__(self):
        with ctxl.ExitStack() as c:
            c.enter_context(register_module_forward_hook(self._fwd))
            c.enter_context(register_module_forward_pre_hook(self._fwd_pre))

            yield self

    def _fwd_pre(self, module: nn.Module, input) -> None:
        "The forward pre-hook."
        self.stack.append(_ModuleInput(module, input))

    def _fwd(self, module: nn.Module, input, output) -> None:
        "The forward hook."
        module_input = self.stack.top()

        # Sanity check to ensure that the modules and inputs are the right ones.
        assert module is module_input.module
        assert input is module_input.input

        self.stack.pop()


@dcls.dataclass(frozen=True)
class ModuleInOutHist:
    """
    The history to track input and output of a `nn.Module` during running.

    Will need to integrate with `Hist` in the future.
    """

    history: list[ModuleThunk] = dcls.field(default_factory=list)
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

    def __getitem__(self, idx: int) -> ModuleThunk:
        return self.history[idx]

    def append(self, thunk: ModuleThunk) -> None:
        outputs = tuple(thunk.downstream())

        # Check if the keys already exists,
        # should be unique due to cloning in fake mode.
        if not self.output_index.keys().isdisjoint(outputs):
            raise KeyError(f"Impossible conflicting output at {self.history[-1]}.")

        length = len(self)
        self.history.append(thunk)

        for output in outputs:
            self.output_index[output] = length

    def thunk_of(self, output: torch.Tensor) -> ModuleThunk:
        idx = self.output_index[output]
        return self[idx]

    def module_forward_hook(
        self, module: nn.Module, input: tuple[torch.Tensor, ...], output: torch.Tensor
    ) -> None:
        thunk = ModuleThunk(
            func=module, args=input, kwargs={}, result=output, parents=()
        )
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


def _is_leaf_module(module: nn.Module) -> bool:
    return not list(module.children())
