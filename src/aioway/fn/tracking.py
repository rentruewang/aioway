# Copyright (c) AIoWay Authors - All Rights Reserved

"Tracking / logging related utilities."

import collections
import dataclasses as dcls
import logging
import typing

import networkx as nx
import rich
import torch

from aioway.schemas import attr

from .faten import FatenFn
from .fn import FnStack
from .guards import TensorFilter, all_tensors, is_leaf_has_grad
from .modes import TDispatchFn, TDispatchMode, TFunctionFn, TFunctionMode

__all__ = [
    "print_torch_function",
    "print_torch_dispatch",
    "LogTorchFunction",
    "LogTorchDispatch",
    "TorchFunctionStack",
    "TorchDispatchStack",
    "FnHistory",
]

LOGGER = logging.getLogger(__name__)


@TFunctionMode.register
def print_torch_function(thunk: TFunctionFn) -> torch.Tensor:
    """
    Print the function calls.
    """

    result = thunk.do()
    rich.print(thunk)
    return result


@TDispatchMode.register
def print_torch_dispatch(thunk: TDispatchFn) -> torch.Tensor:
    """
    Print the dispatcher.
    """

    result = thunk.do()
    rich.print(thunk)
    return result


@dcls.dataclass
class LogTorchFunction(TFunctionMode):
    """
    Log every call to function mode.
    """

    level: int
    "The level to log to."

    logger: logging.Logger = LOGGER
    "The logger to log to. Default to the one in the current module."

    @typing.override
    def __call__(self, thunk: TFunctionFn) -> torch.Tensor:
        result = thunk.do()
        self.logger.log(self.level, "%s", thunk)
        return result


@dcls.dataclass
class LogTorchDispatch(TDispatchMode):
    """
    Log every call to dispatch mode.
    """

    level: int
    "The level to log to."

    logger: logging.Logger = LOGGER
    "The logger to log to. Default to the one in the current module."

    @typing.override
    def __call__(self, thunk: TDispatchFn) -> torch.Tensor:
        result = thunk.do()
        self.logger.log(self.level, "%s", thunk)
        return result


@dcls.dataclass
class TorchFunctionStack(TFunctionMode):
    stack: FnStack[TFunctionFn] = dcls.field(default_factory=FnStack)

    @typing.override
    def __call__(self, thunk: TFunctionFn) -> typing.Any:
        with self.stack.track(thunk):
            return thunk.do()


@dcls.dataclass
class TorchDispatchStack(TDispatchMode):
    stack: FnStack[TDispatchFn] = dcls.field(default_factory=FnStack)

    @typing.override
    def __call__(self, thunk: TDispatchFn) -> torch.Tensor:
        with self.stack.track(thunk):
            return thunk.do()

    def __len__(self) -> int:
        return len(self.stack)

    def top(self) -> TDispatchFn:
        return self.stack.top()


@dcls.dataclass(frozen=True)
class FnResult[F: FatenFn | TFunctionFn | TDispatchFn]:
    fn: F
    result: typing.Any

    @typing.override
    def __repr__(self) -> str:
        return f"{self.fn!r} -> {self.result}"


@dcls.dataclass(frozen=True)
class FnHistory[T: FatenFn | TFunctionFn | TDispatchFn]:
    """
    The list of `Fn` that tracks the current history.
    """

    history: list[FnResult[T]] = dcls.field(default_factory=list)
    """
    The `TorchFn` that has been called, in order.
    """

    input_to_thunk_list: dict[torch.Tensor, list[T]] = dcls.field(
        default_factory=lambda: collections.defaultdict(list)
    )
    "The mapping from input to the thunk containing that input."

    output_to_thunk: dict[torch.Tensor, T] = dcls.field(default_factory=dict)
    "The mapping from output to thunk that generates it."

    def __len__(self) -> int:
        return len(self.history)

    def __getitem__(self, idx: int) -> FnResult[T]:
        return self.history[idx]

    def __iter__(self):
        yield from self.history

    def append(self, item: T, result: typing.Any, /):
        self.history.append(FnResult(item, result))
        self._update_ref(item, result)

    def pop(self):
        return self.history.pop()

    def _update_ref(self, item: T, output: typing.Any) -> None:
        # `__torch_function__` doesn't always return `torch.Tensor` actually!
        if isinstance(output, torch.Tensor):
            # Update output.
            assert output not in self.output_to_thunk
            self.output_to_thunk[output] = item

        # Update input.
        for input_tensor in item.tensors():
            self.input_to_thunk_list[input_tensor].append(item)

    def networkx(self) -> nx.DiGraph[T]:
        graph: nx.DiGraph[T] = nx.DiGraph()
        graph.add_nodes_from(hist.fn for hist in self.history)

        input_thunks = self.input_to_thunk_list
        output_thunks = self.output_to_thunk

        tensors = set(input_thunks.keys()).intersection(output_thunks.keys())

        for tensor in tensors:
            for target_thunk in input_thunks[tensor]:
                _ = graph.add_edge(self.output_to_thunk[tensor], target_thunk)

        return graph

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
