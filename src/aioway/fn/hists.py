# Copyright (c) AIoWay Authors - All Rights Reserved

"History is used with contexts to save previous events, providing tracking."

import collections
import dataclasses as dcls
import itertools
import logging
import typing

import torch

from aioway._common import find_nested_tensors, is_leaf_has_grad
from aioway.schemas import attr

from .fn import TensorInput, cabc

LOGGER = logging.getLogger(__name__)

__all__ = ["History", "HistoryTensorGraph"]


class HashableTensorInput(typing.Hashable, TensorInput, typing.Protocol): ...


@dcls.dataclass(frozen=True)
class FnResult[F]:
    "The storage class per item for `FnHistory`."

    fn: F
    "The `Fn` that has been called."

    result: object
    "The output that `fn` has produced."

    @typing.override
    def __repr__(self) -> str:
        return f"{self.fn!r} -> {self.result}"


@dcls.dataclass(frozen=True)
class History[T: typing.Hashable]:
    """
    `History` is a list storing previous events in order.

    The history list stores past thunks in the order that we received.
    """

    history: list[FnResult[T]] = dcls.field(default_factory=list)
    """
    The `TorchFn` that has been called, in order.
    """

    def __len__(self) -> int:
        return len(self.history)

    def __getitem__(self, idx: int) -> FnResult[T]:
        return self.history[idx]

    def __iter__(self) -> cabc.Generator[FnResult[T]]:
        yield from self.history

    def append(self, thunk: T, result: object, /) -> None:
        "Add a new entry in the `History`."
        self.history.append(FnResult(thunk, result))

    def pop(self) -> FnResult[T]:
        "Drop the last result from the `History`."
        return self.history.pop()


@dcls.dataclass(frozen=True)
class HistoryTensorGraph[T: HashableTensorInput](History[T]):
    """
    `HistoryTensorGraph` is a `History` that can be converted to a graph,
    using the `torch.Tensor`s in the inputs and outputs as links.

    An edge is present if between 2 thunks, if there is a `torch.Tensor`
    both in the input of the first thunk and marked as another thunk's output.

    This stores `torch.Tensor` as `dict` keys, which is fine
    because `torch.Tensor` uses `id` as `hash`,
    and we only care about objects allocated not data equality.
    """

    input_to_thunk_list: dict[torch.Tensor, list[T]] = dcls.field(
        default_factory=lambda: collections.defaultdict(list)
    )
    "The mapping from input to the thunk containing that input."

    output_to_thunk_list: dict[torch.Tensor, list[T]] = dcls.field(
        default_factory=lambda: collections.defaultdict(list)
    )
    "The mapping from output to thunk that generates it."

    @typing.override
    def append(self, thunk: T, result: object, /) -> None:
        super().append(thunk, result)
        self._update_ref(thunk, result)

    def _update_ref(self, thunk: T, output: object) -> None:
        # `__torch_function__` doesn't always return `torch.Tensor` actually!

        # Update output if tensors are found in the output.
        for output_tensor in find_nested_tensors(output):
            self.output_to_thunk_list[output_tensor].append(thunk)

        # Update input.
        for input_tensor in thunk.inputs():
            self.input_to_thunk_list[input_tensor].append(thunk)

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

    def inputs(self) -> set[torch.Tensor]:
        """
        All the inputs of `FnHistory` in a `set`.

        Inputs are defined as tensors not created by operations tracked by `self`.

        Returns:
            A `set` that stores all the `inputs` that are not created by `self`.
        """

        return self._all_inputs() - self._all_outputs()

    def numel(self) -> int:
        "The total number of elements of the tensors."
        return sum(param.numel() for param in self._all_tensors())

    def memory(self) -> int:
        "The total memory consumed by the tensors."
        return sum(attr(param).memory() for param in self._all_tensors())

    def parameters(self):
        for tensor in self._all_tensors():
            if is_leaf_has_grad(tensor):
                yield tensor

    def _all_inputs(self):
        def inputs():
            for hist in self.history:
                yield from hist.fn.inputs()

        return set(inputs())

    def _all_outputs(self):
        def outputs():
            for hist in self.history:
                yield from find_nested_tensors(hist.result)

        return set(outputs())

    def _all_tensors(self):
        return self._all_inputs() | self._all_outputs()
