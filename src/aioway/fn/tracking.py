# Copyright (c) AIoWay Authors - All Rights Reserved

"Tracking / logging related utilities."

import collections
import dataclasses as dcls
import itertools
import logging
import typing

import rich
import torch

from aioway._common import Stack, TensorFilter, filter_tensor_off, is_leaf_has_grad
from aioway.schemas import attr

from .modes import (
    FateFn,
    TDispatchFn,
    TDispatchMode,
    TFunctionFn,
    TFunctionMode,
    replace_tensors_with_attr,
)

__all__ = [
    "PrintTorchFunction",
    "PrintTorchDispatch",
    "LogTorchFunction",
    "LogTorchDispatch",
    "TorchFunctionStack",
    "TorchDispatchStack",
    "FnHistory",
]

LOGGER = logging.getLogger(__name__)


class PrintTorchFunction(TFunctionMode):
    rich: bool = False

    @typing.override
    def __call__(self, thunk: TFunctionFn, /) -> torch.Tensor:
        return _ThunkPrinter(rich=self.rich)(thunk)


class PrintTorchDispatch(TDispatchMode):
    rich: bool = False

    @typing.override
    def __call__(self, thunk: TDispatchFn, /) -> torch.Tensor:
        return _ThunkPrinter(rich=self.rich)(thunk)


@dcls.dataclass(frozen=True)
class _ThunkPrinter:
    rich: bool
    "Use rich for printing."

    def __call__(self, thunk: TFunctionFn | TDispatchFn) -> torch.Tensor:
        self.print("invoke", thunk)
        result = thunk.do()
        self.print("return", thunk, "->", replace_tensors_with_attr(result))
        return result

    @property
    def print(self):
        return rich.print if self.rich else print


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
    stack: Stack[TFunctionFn] = dcls.field(default_factory=Stack)

    @typing.override
    def __call__(self, thunk: TFunctionFn) -> typing.Any:
        with self.stack.enter(thunk):
            return thunk.do()


@dcls.dataclass
class TorchDispatchStack(TDispatchMode):
    stack: Stack[TDispatchFn] = dcls.field(default_factory=Stack)

    @typing.override
    def __call__(self, thunk: TDispatchFn) -> torch.Tensor:
        with self.stack.enter(thunk):
            return thunk.do()

    def __len__(self) -> int:
        return len(self.stack)

    def top(self) -> TDispatchFn:
        return self.stack.top()


@dcls.dataclass(frozen=True)
class FnResult[F: FateFn | TFunctionFn | TDispatchFn]:
    fn: F
    result: typing.Any

    @typing.override
    def __repr__(self) -> str:
        return f"{self.fn!r} -> {self.result}"


@dcls.dataclass(frozen=True)
class FnHistory[T: FateFn | TFunctionFn | TDispatchFn]:
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

    output_to_thunk_list: dict[torch.Tensor, list[T]] = dcls.field(
        default_factory=lambda: collections.defaultdict(list)
    )
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
            self.output_to_thunk_list[output].append(item)

        # Update input.
        for input_tensor in item.tensors():
            self.input_to_thunk_list[input_tensor].append(item)

    def networkx(self):
        import networkx as nx

        graph: nx.DiGraph[T] = nx.DiGraph()
        graph.add_nodes_from(hist.fn for hist in self.history)

        ins = self.input_to_thunk_list
        outs = self.output_to_thunk_list

        tensors = set(ins.keys()).intersection(outs.keys())

        for tensor in tensors:
            for out_thunk, in_thunk in itertools.product(outs[tensor], ins[tensor]):
                _ = graph.add_edge(out_thunk, in_thunk)

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
        return sum(param.numel() for param in self.parameters(filter_tensor_off))

    def memory(self) -> int:
        return sum(attr(param).memory() for param in self.parameters(filter_tensor_off))

    def find_fn(self, tensor: torch.Tensor):
        return self.output_to_thunk_list[tensor]
