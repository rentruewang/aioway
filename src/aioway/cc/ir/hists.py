# Copyright (c) AIoWay Authors - All Rights Reserved

"History is used with contexts to save previous events, providing tracking."

import collections
import dataclasses as dcls
import typing
from collections import abc as cabc

import torch

from aioway._thunks import Thunk
from aioway.torch import (
    Attr,
    ModeThunk,
    find_nested_tensors,
    is_leaf_has_grad,
    replace_tensors_with_attr,
)

from .dags import topo_sort


__all__ = ["Hist", "HistTensorGraph", "TensorInput", "HashableTensorInput"]


@typing.runtime_checkable
class TensorInput(typing.Protocol):
    """
    `TensorInput` marks a class whose value depend on input tensors for computation.
    """

    def inputs(self) -> cabc.Iterable[torch.Tensor]:
        "The tensor operands (inputs to the function)"

        raise NotImplementedError


class HashableTensorInput(typing.Hashable, TensorInput, Thunk, typing.Protocol): ...


@typing.runtime_checkable
class TensorNode(TensorInput, Thunk, typing.Protocol):
    f"""
    `TensorNode` have both tensor output (`run()`) and tensor inputs (`.inputs()`).
    The output itself does not need to be tensor, but must decompose (only) into tensors.
    """


@dcls.dataclass(frozen=True)
class ThunkResult[F]:
    "The storage class per item for `FnHistory`."

    thunk: F
    "The `Thunk` that has been called."

    result: object
    "The output that `fn` has produced."

    @typing.override
    def __repr__(self) -> str:
        if isinstance(self.result, Exception):
            return f"{type(self.result).__name__}: {self.result}: {self.thunk!r}"
        else:
            result = replace_tensors_with_attr(self.result)
            return f"{self.thunk!r} -> {result}"


@dcls.dataclass(frozen=True)
class Hist[T: Thunk]:
    """
    `Hist` is a list storing previous events in order.

    The history list stores past thunks in the order that we received.
    """

    history: list[ThunkResult[T]] = dcls.field(default_factory=list)
    """
    The `ModeThunk` that has been called, in order.
    """

    def __bool__(self) -> bool:
        return bool(len(self))

    def __len__(self) -> int:
        return len(self.history)

    def __getitem__(self, idx: int) -> ThunkResult[T]:
        return self.history[idx]

    def __iter__(self) -> cabc.Generator[ThunkResult[T]]:
        yield from self.history

    def __repr__(self) -> str:
        return repr(self.history)

    def execute(self, thunk: T) -> object:
        try:
            result = thunk()
        except Exception as e:
            self._append(thunk, e)
            raise
        else:
            self._append(thunk, result)
            return result

    def _append(self, thunk: T, result: object | Exception, /) -> None:
        "Add a new entry in the `History`."
        self.history.append(ThunkResult(thunk, result))


@dcls.dataclass(frozen=True)
class HistTensorGraph[T: ModeThunk | HashableTensorInput](Hist):
    """
    `HistTensorGraph` is a `Hist` that can be converted to a graph,
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
    def _append(self, thunk: T, result: object, /) -> None:
        super()._append(thunk, result)
        self._update_ref(thunk, result)

    def _update_ref(self, thunk: T, output: object) -> None:
        # `__torch_function__` doesn't always return `torch.Tensor` actually!

        # Update output if tensors are found in the output.
        for output_tensor in find_nested_tensors(output):
            self.output_to_thunk_list[output_tensor].append(thunk)

        # Update input.
        for input_tensor in thunk.inputs():
            self.input_to_thunk_list[input_tensor].append(thunk)

    def topo_sort(self) -> list[T]:
        "Sort the tensor graph topologically."

        graph = self._thunk_graph()
        return topo_sort(graph)

    def _thunk_graph(self):
        outs = self.output_to_thunk_list

        # From input -> someone's output -> map to thunk itself.
        def input_thunks(thunk: T):
            for tensor in thunk.inputs():
                for input_thunk in outs[tensor]:
                    yield input_thunk

        return {entry.thunk: list(input_thunks(entry.thunk)) for entry in self.history}

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
        return sum(Attr.parse(param).memory() for param in self._all_tensors())

    def parameters(self):
        for tensor in self._all_tensors():
            if is_leaf_has_grad(tensor):
                yield tensor

    def _all_inputs(self) -> set[torch.Tensor]:
        def inputs():
            for hist in self.history:
                yield from hist.thunk.inputs()

        return set(inputs())

    def _all_outputs(self) -> set[torch.Tensor]:
        def outputs():
            for hist in self.history:
                yield from find_nested_tensors(hist.result)

        return set(outputs())

    def _all_tensors(self) -> set[torch.Tensor]:
        return self._all_inputs() | self._all_outputs()
