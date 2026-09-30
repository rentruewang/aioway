# Copyright (c) AIoWay Authors - All Rights Reserved

"The nodes and edges for DAG."

import dataclasses as dcls
import functools
import typing
from collections import abc as cabc

import torch

from aioway._utils import AnySet, any_set
from aioway.t import (
    TList,
    find_nested_tensors,
    is_real,
    render_tensor_func_short,
    replace_tensors_with_attr,
)

__all__ = ["ThunkNode", "TensorRef"]


@dcls.dataclass(frozen=True, eq=False, repr=False)
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

    def __eq__(self, other) -> bool:
        """
        `ThunkNode` will not implement `__eq__`.
        """
        return NotImplemented

    @typing.override
    def __repr__(self) -> str:
        result = str(replace_tensors_with_attr(self.result))
        thunk = render_tensor_func_short(str(self.func), self.args, self.kwargs)
        return thunk + " -> " + result

    @functools.cached_property
    def inputs(self) -> TList:
        "Get the (unique) dependencies of the current thunk."
        return TList(self._in_tensors())

    @functools.cached_property
    def outputs(self) -> TList:
        "Get the output list of (unique) tensors of the current thunk."
        return TList(self._out_tensors())

    def _in_tensors(self) -> cabc.Generator[torch.Tensor]:
        yield from find_nested_tensors(self.args)
        yield from find_nested_tensors(self.kwargs)

    def _out_tensors(self) -> cabc.Generator[torch.Tensor]:
        yield from find_nested_tensors(self.result)

    @property
    def done(self) -> bool:
        return self.result is not dcls.MISSING


class TensorRef:
    """
    A data structure holding tensor information in the DAG.

    It holds the reference to the fake tensor,
    the callable that produces it, and the downstream consumer (function).

    Since in the DAG the tensor are never reused by the function,
    we can assume there is only 1 single producer.
    """

    def __init__(self, producer: ThunkNode | None, tensor: torch.Tensor) -> None:
        """
        Args:
            producer: The thunk's id, or `None` if it's a free variable.
            tensor: The tensor this reference tracks. Must be fake.
        """

        self._producer = producer
        self._tensor = tensor
        self._consumers = any_set(ThunkNode)

        if not isinstance(tensor, torch.Tensor) or is_real(tensor):
            raise ValueError(
                f"The fake tensor produced at idx={self._producer} is real."
            )

    def add_consumers(self, *consumers: ThunkNode) -> None:
        "Add consumers for the info. Allow duplication."

        for consumer in consumers:
            self._consumers.add(consumer)

    @property
    def producer(self) -> ThunkNode:
        "The producer index."

        if self._producer is None:
            raise AttributeError("The variable is a free variable.")
        else:
            return self._producer

    @property
    def consumers(self) -> AnySet[ThunkNode]:
        "The list of consumers."
        return self._consumers

    @property
    def tensor(self) -> torch.Tensor:
        "Return the fake tensor."
        return self._tensor

    @property
    def is_free(self) -> bool:
        "Check if the tensor is a free value."
        return self._producer is None
