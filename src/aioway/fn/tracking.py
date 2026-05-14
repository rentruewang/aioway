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
from aioway._common.decomps import find_nested_tensors
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

type TorchCall = FateFn | TFunctionFn | TDispatchFn
"The calls to `torch.*` APIs."


class PrintTorchFunction(TFunctionMode):
    rich: bool = False

    @typing.override
    def __call__(self, thunk: TFunctionFn, /) -> object:
        return _ThunkPrinter(rich=self.rich)(thunk)


class PrintTorchDispatch(TDispatchMode):
    rich: bool = False

    @typing.override
    def __call__(self, thunk: TDispatchFn, /) -> object:
        return _ThunkPrinter(rich=self.rich)(thunk)


@dcls.dataclass(frozen=True)
class _ThunkPrinter:
    rich: bool
    "Use rich for printing."

    def __call__(self, thunk: TFunctionFn | TDispatchFn) -> object:
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
    def __call__(self, thunk: TFunctionFn) -> object:
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
    def __call__(self, thunk: TDispatchFn) -> object:
        result = thunk.do()
        self.logger.log(self.level, "%s", thunk)
        return result


@dcls.dataclass
class TorchFunctionStack(TFunctionMode):
    stack: Stack[TFunctionFn] = dcls.field(default_factory=Stack)

    @typing.override
    def __call__(self, thunk: TFunctionFn) -> object:
        with self.stack.enter(thunk):
            return thunk.do()


@dcls.dataclass
class TorchDispatchStack(TDispatchMode):
    stack: Stack[TDispatchFn] = dcls.field(default_factory=Stack)

    @typing.override
    def __call__(self, thunk: TDispatchFn) -> object:
        with self.stack.enter(thunk):
            return thunk.do()

    def __len__(self) -> int:
        return len(self.stack)

    def top(self) -> TDispatchFn:
        return self.stack.top()


@dcls.dataclass(frozen=True)
class FnResult[F: TorchCall]:
    "The storage class per item for `FnHistory`."

    fn: F
    "The `Fn` that has been called."

    result: object
    "The output that `fn` has produced."

    @typing.override
    def __repr__(self) -> str:
        return f"{self.fn!r} -> {self.result}"


@dcls.dataclass(frozen=True)
class FnHistory[T: TorchCall]:
    """
    The list of `Fn` that tracks the current history.

    This stores `torch.Tensor` as `dict` keys, which is fine
    because `torch.Tensor` uses `id` as `hash`,
    and we only care about objects allocated not data equality.
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

    def append(self, item: T, result: object, /):
        self.history.append(FnResult(item, result))
        self._update_ref(item, result)

    def pop(self):
        return self.history.pop()

    def _update_ref(self, item: T, output: object) -> None:
        # `__torch_function__` doesn't always return `torch.Tensor` actually!

        # Update output if tensors are found in the output.
        for output_tensor in find_nested_tensors(output):
            self.output_to_thunk_list[output_tensor].append(item)

        # Update input.
        for input_tensor in item.tensors():
            self.input_to_thunk_list[input_tensor].append(item)

    def networkx(self):
        "Convert the graph to `nx.DiGraph`, using data dependencies as link."

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

    def inputs(self) -> set[torch.Tensor]:
        """
        All the inputs of `FnHistory` in a `set`.

        Inputs are defined as tensors not created by operations tracked by `self`.

        Returns:
            A `set` that stores all the `inputs` that are not created by `self`.
        """

        def all_inputs():
            for hist in self.history:
                yield from hist.fn.tensors()

        def all_outputs():
            for hist in self.history:
                yield from find_nested_tensors(hist.result)

        inputs = set(all_inputs())
        outputs = set(all_outputs())

        return inputs - outputs

    def numel(self) -> int:
        "The total number of elements of the tensors."
        return sum(param.numel() for param in self.parameters(filter_tensor_off))

    def memory(self) -> int:
        "The total memory consumed by the tensors."
        return sum(attr(param).memory() for param in self.parameters(filter_tensor_off))
