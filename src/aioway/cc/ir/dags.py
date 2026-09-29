# Copyright (c) AIoWay Authors - All Rights Reserved

"The DAG that supports analysis."

import collections
import dataclasses as dcls
import functools
import itertools
import typing
from collections import abc as cabc

import torch

from aioway.t import (
    TList,
    find_nested_tensors,
    is_real,
    render_tensor_func_short,
    replace_tensors_with_attr,
)

__all__ = ["ThunkNode", "TensorRef", "Dag"]


@dcls.dataclass(frozen=True)
class ThunkNode[F: cabc.Callable]:
    """
    Stores the thunk's arguments, function, and output.

    This is the node type for the dag.
    """

    _: dcls.KW_ONLY

    func: F
    "The callable that the thunk calls."

    args: tuple[typing.Any, ...]
    "The arguments fed to the function."

    kwargs: dict[str, typing.Any]
    "The keyword arguments fed to the function."

    result: typing.Any = dcls.MISSING
    "The result. If `dcls.MISSING`, the thunk is not called yet."

    def __post_init__(self) -> None:
        if not callable(self.func):
            raise TypeError(f"{self.func} is not callable.")

    @typing.override
    def __repr__(self) -> str:
        result = str(replace_tensors_with_attr(self.result))
        thunk = render_tensor_func_short(str(self.func), self.args, self.kwargs)
        return thunk + " -> " + result

    @functools.cached_property
    def inputs(self) -> TList:
        "Get the (unique) dependencies of the current thunk."
        return TList(self._upstream())

    @functools.cached_property
    def outputs(self) -> TList:
        "Get the output list of (unique) tensors of the current thunk."
        return TList(self._downstream())

    def _upstream(self) -> cabc.Generator[torch.Tensor]:
        yield from find_nested_tensors(self.args)
        yield from find_nested_tensors(self.kwargs)

    def _downstream(self) -> cabc.Generator[torch.Tensor]:
        yield from find_nested_tensors(self.result)

    @property
    def done(self) -> bool:
        return self.result is not dcls.MISSING


class TensorRef[T: cabc.Callable = cabc.Callable]:
    """
    A data structure holding tensor information in the DAG.

    It holds the reference to the fake tensor,
    the callable that produces it, and the downstream consumer (function).

    Since in the DAG the tensor are never reused by the function,
    we can assume there is only 1 single producer.
    """

    def __init__(self, producer: T | None, fake: torch.Tensor):
        self._producer = producer
        self._fake = fake
        self._consumers: set[cabc.Callable] = set()

        if not isinstance(fake, torch.Tensor) or is_real(fake):
            raise ValueError(
                f"The fake tensor produced at idx={self._producer} is real."
            )

    def __hash__(self) -> int:
        return id(self.fake)

    def add_consumers(self, *consumers: cabc.Callable) -> None:
        "Add consumers for the info. Allow duplication."

        for consumer in consumers:
            self._consumers.add(consumer)

    @property
    def producer(self) -> T:
        "The producer index."
        if self._producer is None:
            raise AttributeError("The variable is a free variable.")

        return self._producer

    @property
    def consumers(self) -> cabc.Set[cabc.Callable]:
        "The list of consumers."
        return self._consumers

    @property
    def fake(self) -> torch.Tensor:
        "Return the fake tensor."
        return self._fake

    @property
    def is_free(self) -> bool:
        "Check if the tensor is a free value."
        return self._producer is None


class Dag[F: cabc.Callable]:
    """
    A dag is a sequence of callables, that are linked by fake tensors.
    """

    def __init__(
        self,
        thunks: cabc.Iterable[ThunkNode[F]],
        inputs: cabc.Iterable[torch.Tensor],
        outputs: cabc.Iterable[torch.Tensor],
    ) -> None:
        self._thunks = tuple(thunks)

        self._inputs = tuple(inputs)
        self._outputs = tuple(outputs)

        self._func_index = {thunk.func: i for i, thunk in enumerate(self._thunks)}
        "Mapping from function to step."

        if len(self._func_index) != len(self._thunks):
            raise ValueError("The function list in thunks is not unique.")

        self._tensors = self._build_tensor_refs()
        "Mapping from tensors to refs (linking functions)."

        self._inputs_to_step = self._step_input_mapping()
        "The mapping from tensor id to step that uses it."

        # Validate if the inputs and outputs are valid.
        self._validate_input_output()

        if not self.tensors.all_fake:
            raise ValueError("Contains non fake tensors.")

    def __len__(self) -> int:
        return len(self._thunks)

    def __iter__(self) -> cabc.Iterator[ThunkNode[F]]:
        return iter(self._thunks)

    def __getitem__(self, idx: int):
        return self._thunks[idx]

    def first_use(self, tensor: torch.Tensor) -> int:
        return self.__get_tensor_life(tensor, min)

    def last_use(self, tensor: torch.Tensor) -> int:
        return self.__get_tensor_life(tensor, max)

    @property
    def inputs(self):
        return self._inputs

    @property
    def outputs(self):
        return self._outputs

    @functools.cached_property
    def tensors(self) -> TList:
        return TList(self._all_tenors())

    def func_step_index(self, func: F) -> int:
        return self._func_index[func]

    def output_of_step(self, tensor: torch.Tensor) -> int:
        """
        Get the step number of step that produced output.

        If not set (producer is None), return -1.
        """

        ref = self._tensors[id(tensor)]

        try:
            func = ref.producer
        except AttributeError:
            return -1
        else:
            return self._func_index[func]

    def input_to_step(self, tensor: torch.Tensor) -> cabc.Sequence[int]:
        return self._inputs_to_step[id(tensor)]

    def __get_all_tensors_produced(self):
        for i, thunk in enumerate(self._thunks):
            for tensor in thunk.outputs:
                yield i, tensor

    def _step_output_mapping(self) -> dict[int, int]:
        idx_to_tensors = list(self.__get_all_tensors_produced())

        result = {}
        for i, tensor in idx_to_tensors:
            result[id(tensor)] = i

        if len(result) != len(idx_to_tensors):
            raise ValueError("Output tensors of thunks are not unique.")

        return result

    def _all_tenors(self):
        yield from self._inputs
        yield from self._outputs

        yield from _all_thunk_tensors(self._thunks)

    def _step_input_mapping(self) -> dict[int, list[int]]:
        result: dict[int, list[int]] = collections.defaultdict(list)

        for i, thunk in enumerate(self._thunks):
            for tensor in thunk.inputs:
                result[id(tensor)].append(i)

        return result

    def _validate_input_output(self) -> None:
        for input in self._inputs:
            # Input should not have a producer.
            if not self._tensors[id(input)].is_free:
                raise ValueError("Input is not a free variable.")

            # Input should be used.
            if id(input) not in self._inputs_to_step:
                raise ValueError("Input is not used.")

        for output in self._outputs:
            # Output is not produced.
            if id(output) not in self._tensors:
                raise ValueError("Output not discovered.")

        if set(self._inputs) & set(self._outputs):
            raise ValueError("Inputs are in the outputs. Not allowed yet.")

    def __get_tensor_life(
        self, tensor: torch.Tensor, func: cabc.Callable[[cabc.Iterable[int]], int]
    ) -> int:
        # Use the ids to check because tensor `==` compares by element.

        if id(tensor) in self._input_ids:
            return -1

        if id(tensor) in self._output_ids:
            return len(self)

        return func(self._inputs_to_step[id(tensor)])

    @functools.cached_property
    def _input_ids(self) -> frozenset[int]:
        return frozenset(id(t) for t in self._inputs)

    @functools.cached_property
    def _output_ids(self) -> frozenset[int]:
        return frozenset(id(t) for t in self._outputs)

    def _build_tensor_refs(self) -> dict[int, TensorRef[F]]:
        mapping: dict[int, TensorRef[F]] = {}

        # Register all inputs.
        for input in self.inputs:
            mapping[id(input)] = TensorRef(producer=None, fake=input)

        # Register all outputs of thunks.
        for thunk in self._thunks:
            for output in thunk.outputs:
                if id(output) in mapping:
                    raise ValueError("The output fake tensor is not unique.")

                mapping[id(output)] = TensorRef(producer=thunk.func, fake=output)

        # Register each thunk's input.
        for thunk in self._thunks:
            for input in thunk.inputs:
                assert id(input) in mapping
                mapping[id(input)].add_consumers(thunk.func)

        return mapping

    @classmethod
    def from_thunk_list(cls, thunks: cabc.Sequence[ThunkNode[F]]) -> typing.Self:
        """
        Given only the thunk list, construct a DAG, auto discover inputs and outputs.

        Inputs = unproduced tensors that exists in graph.
        Outputs = unused tensors in graph.
        """

        # Using ordered dicts to preserve insertion order, which affects signature.
        inputs: dict[int, _TensorStamp] = {}
        outputs: dict[int, _TensorStamp] = {}

        counter = itertools.count()

        for thunk in thunks:
            for i in thunk.inputs:
                inputs[id(i)] = _TensorStamp(next(counter), i)

            for o in thunk.outputs:
                outputs[id(o)] = _TensorStamp(next(counter), o)

        input_only = inputs.keys() - outputs.keys()
        output_only = outputs.keys() - inputs.keys()

        input_tensors = tuple(t for _, t in sorted(inputs[k] for k in input_only))
        output_tensors = tuple(t for _, t in sorted(outputs[k] for k in output_only))

        return cls(thunks, input_tensors, output_tensors)


class _TensorStamp(typing.NamedTuple):
    stamp: int
    tensor: torch.Tensor


def _all_thunk_tensors(thunks: cabc.Sequence[ThunkNode]):
    for thunk in thunks:
        yield from thunk.inputs
        yield from thunk.outputs
