# Copyright (c) AIoWay Authors - All Rights Reserved

import contextlib as ctxl
import dataclasses as dcls
import typing
from collections import abc as cabc

import torch
from torch import nn

from aioway._utils import AnyDict, Stack, any_dict
from aioway.ir import InstrList
from aioway.t import (
    register_module_forward_hook,
    register_module_forward_pre_hook,
)

from .instrs import ModuleCall
from .progs import Program

__all__ = ["ModuleTracker", "ModuleHist", "track_module_thunks"]


@ctxl.contextmanager
def track_module_thunks() -> cabc.Generator[ModuleHist]:
    tracker = ModuleTracker()

    with tracker():
        yield tracker.hist


class _ModuleInput(typing.NamedTuple):
    module: nn.Module
    input: typing.Any


@dcls.dataclass(frozen=True)
class ModuleHist:
    """
    The history to track input and output of a `nn.Module` during running.

    Will need to integrate with `Hist` in the future.
    """

    history: list[ModuleCall] = dcls.field(default_factory=list)
    """
    The history encountered.
    """

    output_index: AnyDict[torch.Tensor, int] = dcls.field(default_factory=any_dict)
    """
    The index of each tensor where `thunk.output = tensor`.

    Since this graph is about expression, we only care about the output.
    """

    def __len__(self) -> int:
        return len(self.history)

    def __getitem__(self, idx: int) -> ModuleCall:
        return self.history[idx]

    def append(self, thunk: ModuleCall) -> None:
        outputs = tuple(thunk.outputs)

        # Check if the keys already exists,
        # should be unique due to cloning in fake mode.
        if not self.output_index.keys().isdisjoint(outputs):
            raise KeyError(f"Impossible conflicting output at {self.history[-1]}.")

        length = len(self)
        self.history.append(thunk)

        for output in outputs:
            self.output_index[output] = length

    def thunk_of(self, output: torch.Tensor) -> ModuleCall:
        idx = self.output_index[output]
        return self[idx]

    def module_forward_hook(
        self, module: nn.Module, input: tuple[torch.Tensor, ...], output: torch.Tensor
    ) -> None:
        thunk = ModuleCall(
            func=module, args=input, kwargs={}, result=output, parents=()
        )
        self.append(thunk)

    @property
    def instrs(self) -> InstrList[ModuleCall]:
        return InstrList.build(self.history)

    @property
    def program(self) -> Program[ModuleCall]:
        return Program.from_instr_list(self.history)


@dcls.dataclass(frozen=True)
class ModuleTracker:
    stack: Stack[_ModuleInput] = dcls.field(default_factory=Stack)
    "The current stack of modules."

    hist: ModuleHist = dcls.field(default_factory=ModuleHist)
    "The history of modules execution (only leaves are tracked)."

    @ctxl.contextmanager
    def __call__(self):
        with ctxl.ExitStack() as c:
            c.enter_context(register_module_forward_hook(self._fwd_post))
            c.enter_context(register_module_forward_pre_hook(self._fwd_pre))

            yield self

    def _fwd_pre(self, module: nn.Module, input) -> None:
        "The forward pre-hook."
        self.stack.append(_ModuleInput(module, input))

    def _fwd_post(self, module: nn.Module, input, output) -> None:
        "The forward hook (this executes after)."

        self._fwd_post_hist(module, input, output)
        self._fwd_post_stack(module, input)

    def _fwd_post_hist(self, module: nn.Module, input, output):
        """
        Update the history. This executes "before" the stack is popped,
        which means that `module` does exist in `self.stack`.

        Only append "leaf" modules, which means modules without parents,
        to prevent double counting.
        """

        assert isinstance(input, tuple)
        assert self.stack.top().module is module

        if not _is_leaf_module(module):
            return

        self.hist.append(
            ModuleCall(
                func=module,
                args=input,
                kwargs={},
                result=output,
                parents=tuple(module for module, _ in self.stack),
            )
        )

    def _fwd_post_stack(self, module: nn.Module, input):
        module_input = self.stack.top()

        # Sanity check to ensure that the modules and inputs are the right ones.
        assert module is module_input.module
        assert input is module_input.input

        self.stack.pop()


def _is_leaf_module(module: nn.Module) -> bool:
    return not list(module.children())
