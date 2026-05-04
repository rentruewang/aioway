# Copyright (c) AIoWay Authors - All Rights Reserved

"Tracking / logging related utilities."

import dataclasses as dcls
import logging
import typing
from collections import abc as cabc

import torch
from torch import _ops

from aioway.fn.modes import TorchDispatchFn
from aioway.schemas import attr

from .fn import Fn, FnStack, Thunk
from .guards import TensorFilter, all_tensors, is_leaf_has_grad
from .modes import (
    TorchDispatchFn,
    TorchDispatchMode,
    TorchFunctionFn,
    TorchFunctionMode,
)
from .previews import TensorFn

__all__ = [
    "print_torch_dispatch",
    "LogTorchDispatch",
    "TorchFunctionStack",
    "TorchDispatchStack",
    "FnHistory",
    "DispatchHistory",
]

LOGGER = logging.getLogger(__name__)


@TorchDispatchMode.register
def print_torch_dispatch(
    func: _ops.OpOverload,
    types: tuple[type[torch.Tensor], ...],
    args: tuple[typing.Any, ...],
    kwargs: dict[str, typing.Any],
):
    """
    Print the dispatcher.
    """
    invoke = Thunk(func, *args, **kwargs)

    result = invoke.do()
    print(invoke)
    return result


@dcls.dataclass
class LogTorchDispatch(TorchDispatchMode):
    """
    Log every call to dispatch mode.
    """

    level: int
    "The level to log to."

    logger: logging.Logger = LOGGER
    "The logger to log to. Default to the one in the current module."

    @typing.override
    def __call__(
        self,
        op: _ops.OpOverload,
        types: tuple[type[torch.Tensor], ...],
        *args: typing.Any,
        **kwargs: typing.Any,
    ) -> torch.Tensor:
        invoke = Thunk(op, *args, **kwargs)
        result = invoke.do()
        self.logger.log(self.level, "%s", invoke)
        return result


@dcls.dataclass
class TorchFunctionStack(TorchFunctionMode):
    stack: FnStack[TorchFunctionFn] = dcls.field(default_factory=FnStack)

    @typing.override
    def __call__(
        self,
        func: cabc.Callable[..., typing.Any],
        types: tuple[type, ...],
        *args: typing.Any,
        **kwargs: typing.Any,
    ) -> typing.Any:
        thunk = TorchFunctionFn(func, types, args, kwargs)
        with self.stack.track(thunk):
            return thunk.do()


@dcls.dataclass
class TorchDispatchStack(TorchDispatchMode):
    stack: FnStack[TorchDispatchFn] = dcls.field(default_factory=FnStack)

    @typing.override
    def __call__(
        self,
        op: _ops.OpOverload,
        types: tuple[type[torch.Tensor], ...],
        *args: typing.Any,
        **kwargs: typing.Any,
    ) -> torch.Tensor:
        thunk = TorchDispatchFn(op, types, args, kwargs)
        with self.stack.track(thunk):
            print(" stack before")
            result = thunk.do()
            print(" stack after")
            return result

    def __len__(self) -> int:
        return len(self.stack)

    def top(self) -> TorchDispatchFn:
        return self.stack.top()


@dcls.dataclass(frozen=True)
class FnResult[F: Fn]:
    fn: F
    result: torch.Tensor


@dcls.dataclass(frozen=True)
class FnHistory[T: TensorFn]:
    """
    The list of `Fn` that tracks the current history.
    """

    history: list[FnResult[T]] = dcls.field(default_factory=list)
    """
    The `TorchFn` that has been called, in order.
    """

    input_to_thunk: dict[torch.Tensor, T] = dcls.field(default_factory=dict)
    "The mapping from input to the thunk containing that input."

    output_to_thunk: dict[torch.Tensor, T] = dcls.field(default_factory=dict)
    "The mapping from output to thunk that generates it."

    thunk_index: dict[int, int] = dcls.field(default_factory=dict)
    """
    The reverse of history, hash of thunk to index of thunk.
    """

    edges: list[tuple[int, int]] = dcls.field(default_factory=list)
    """
    The edges between 2 elements in the history.
    """

    def __len__(self) -> int:
        return len(self.history)

    def __getitem__(self, idx: int) -> FnResult[T]:
        return self.history[idx]

    def __iter__(self):
        yield from self.history

    def append(self, item: T, result: torch.Tensor, /):
        self.history.append(FnResult(item, result))
        self._update_ref(item, result)

    def pop(self):
        return self.history.pop()

    def _update_ref(self, item: T, output: torch.Tensor):
        # Update index.
        curr_idx = len(self.history) - 1
        self.thunk_index[id(item)] = curr_idx

        # Update output.
        assert output not in self.output_to_thunk
        self.output_to_thunk[output] = item

        # Update input.
        for input_tensor in item.tensors():
            assert input_tensor not in self.input_to_thunk
            self.input_to_thunk[input_tensor] = item

            if input_tensor in self.output_to_thunk:
                thunk = self.output_to_thunk[input_tensor]
                idx = self.thunk_index[id(thunk)]
                self.edges.append((idx, curr_idx))


class DispatchHistory(FnHistory[TensorFn]):
    @typing.override
    def _update_ref(self, item: TensorFn, output: torch.Tensor):
        # Here, in dispatch mode, we may deal with non tensor outputs.
        if isinstance(output, torch.Tensor):
            super()._update_ref(item, output)

    def parameters(self, select: TensorFilter = is_leaf_has_grad, unique: bool = True):
        def data_params():
            for result in self.history:
                yield from result.fn.parameters(select)

        params = data_params()

        if unique:
            params = set(data_params())

        yield from params

    def numel(self) -> int:
        return sum(param.numel() for param in self.parameters(all_tensors))

    def memory(self) -> int:
        return sum(attr(param).memory() for param in self.parameters(all_tensors))

    def find_fn(self, tensor: torch.Tensor):
        return self.output_to_thunk[tensor]
