# Copyright (c) AIoWay Authors - All Rights Reserved

"History is used with contexts to save previous events, providing tracking."

import collections
import dataclasses as dcls
import logging
import typing
from collections import abc as cabc

import torch

from aioway._torch import is_leaf_has_grad
from aioway._utils import Dag, TupleDagNode, find_nested_tensors
from aioway.fn import Fn, TensorInput
from aioway.schemas import Attr

from .common import replace_tensors_with_attr

LOGGER = logging.getLogger(__name__)

__all__ = ["Hist", "HistTensorGraph"]


class HashableTensorInput(typing.Hashable, TensorInput, Fn, typing.Protocol): ...


@dcls.dataclass(frozen=True)
class FnResult[F]:
    "The storage class per item for `FnHistory`."

    fn: F
    "The `Fn` that has been called."

    result: object
    "The output that `fn` has produced."

    @typing.override
    def __repr__(self) -> str:
        if isinstance(self.result, Exception):
            return f"{type(self.result).__name__}: {self.result}: {self.fn!r}"
        else:
            result = replace_tensors_with_attr(self.result)
            return f"{self.fn!r} -> {result}"


@dcls.dataclass(frozen=True)
class Hist[T: Fn]:
    """
    `Hist` is a list storing previous events in order.

    The history list stores past thunks in the order that we received.
    """

    history: list[FnResult[T]] = dcls.field(default_factory=list)
    """
    The `TorchFn` that has been called, in order.
    """

    def __bool__(self) -> bool:
        return bool(len(self))

    def __len__(self) -> int:
        return len(self.history)

    def __getitem__(self, idx: int) -> FnResult[T]:
        return self.history[idx]

    def __iter__(self) -> cabc.Generator[FnResult[T]]:
        yield from self.history

    def __repr__(self) -> str:
        return repr(self.history)

    def execute(self, thunk: T) -> object:
        try:
            result = thunk.do()
        except Exception as e:
            self._append(thunk, e)
            raise
        else:
            self._append(thunk, result)
            return result

    def _append(self, thunk: T, result: object | Exception, /) -> None:
        "Add a new entry in the `History`."
        self.history.append(FnResult(thunk, result))


@dcls.dataclass(frozen=True)
class HistTensorGraph[T: HashableTensorInput](Hist[T]):
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

    def dag(self) -> Dag[T]:
        "Conver the graph to a `Dag` for inspection and debugging."

        thunk_idxs = {thunk.fn: idx for idx, thunk in enumerate(self.history)}
        nodes: list[TupleDagNode[T]] = []

        for entry in self.history:
            nodes.append(
                self.__get_dag_thunk(
                    thunk=entry.fn,
                    thunk_idxs=thunk_idxs,
                    nodes=nodes,
                )
            )

        return Dag.from_outputs(nodes)

    def __get_dag_thunk(
        self, *, thunk: T, thunk_idxs: dict[T, int], nodes: list[TupleDagNode[T]]
    ):
        outs = self.output_to_thunk_list
        input_indices: list[int] = []

        # From input -> someone's output -> map to thunk itself -> index.
        def input_idxs():
            for tensor in thunk.inputs():
                for input_fn in outs[tensor]:
                    idx = thunk_idxs[input_fn]
                    yield idx

        # Since `history` is by definition a topologically sorted array,
        # the dependencies would always have smaller index (won't index OOB).
        for in_idx in input_idxs():
            assert in_idx < len(nodes)
            input_indices.append(in_idx)

        deps = tuple(nodes[idx] for idx in input_indices)
        return TupleDagNode(thunk, deps)

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
