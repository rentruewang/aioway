# Copyright (c) AIoWay Authors - All Rights Reserved

import abc
import dataclasses as dcls
import inspect
import re
import typing
from collections import abc as cabc

import torch
from torch import _ops

from aioway._common.dcls import dcls_no_repr
from aioway.fn.funcs import torch_function_stack

from .fn import Fn, Thunk, pretty_function_call
from .guards import TensorFilter, all_tensors

__all__ = ["find_preview", "all_previews", "TorchFn", "Preview", "TorchDispatchThunk"]


_PREVIEW_CANDIDATES: dict[_ops.OpOverload, list[cabc.Callable[..., Preview]]] = {}


class TorchFn(Fn, abc.ABC):
    def __init__(self):
        try:
            self._torch_function = torch_function_stack().top()
        except IndexError:
            raise RuntimeError("There are no `__torch_function__` calls active!")

    @abc.abstractmethod
    @typing.override
    def do(self) -> torch.Tensor:
        raise NotImplementedError

    def parameters(self, select: TensorFilter = all_tensors, /):
        for tensor in self.tensors():
            if select(tensor):
                yield tensor

    @abc.abstractmethod
    def tensors(self) -> cabc.Iterator[torch.Tensor]:
        raise NotImplementedError


class TorchDispatchThunk(Thunk, TorchFn):
    def __init__(
        self,
        func: cabc.Callable[..., typing.Any],
        types: tuple[type, ...],
        /,
        *args: typing.Any,
        **kwargs: typing.Any,
    ) -> None:
        Thunk.__init__(self, func, *args, **kwargs)
        TorchFn.__init__(self)
        self._types = types

    @typing.override
    def __eq__(self, other: object) -> bool:
        if isinstance(other, TorchDispatchThunk):
            return (
                True
                and Thunk.__eq__(self, other)
                and self.types == other.types
                and self._torch_function == other._torch_function
            )

        if isinstance(other, Thunk):
            return other == self

        return NotImplemented

    @property
    @typing.override
    @typing.no_type_check
    def func(self) -> _ops.OpOverload:
        return self._func

    @property
    def types(self):
        return self._types

    @typing.override
    def tensors(self):

        def all_args():
            yield from self.args
            yield from self.kwargs.values()

        for arg in all_args():
            if isinstance(arg, torch.Tensor):
                yield arg


@dcls_no_repr
class Preview(TorchFn, abc.ABC):
    """
    `Preview` is a preview for operations,
    allowing for multiple implementations for the same torch IR.
    """

    IR: typing.ClassVar[_ops.OpOverload]
    """
    The torch IR that this `Preview` would be implementing.
    """

    def __init_subclass__(cls) -> None:
        cls.__register_preview()

    def __post_init__(self):
        TorchFn.__init__(self)

    @typing.override
    def __repr__(self) -> str:
        return pretty_function_call(f"preview::{self.name()}", **dcls.asdict(self))

    @abc.abstractmethod
    def ok(self) -> bool:
        """
        Whether or not the arguments are valid.
        """

        raise NotImplementedError

    @abc.abstractmethod
    def do(self) -> torch.Tensor:
        """
        Generate the fake tensor.
        """

        raise NotImplementedError

    @abc.abstractmethod
    def cost(self) -> int:
        """
        Return the cost of each operation.
        """

        raise NotImplementedError

    @classmethod
    def __register_preview(cls):
        """
        Register a patching function that only runs under fake mode.

        The patch would be called. If the patching function returns `NotImplemented`,
        it will fall back to the default implementation (plain `func(*args, **kwargs)`).
        """

        # Abstract methods.
        if inspect.isabstract(cls):
            return

        # Abstract `ClassVar`.
        try:
            op = cls.IR
        except AttributeError:
            return

        # Mimick defaultdict behavior.
        # Using dict over defaultdict s.t. we don't need special handling in `repr`.
        if op not in _PREVIEW_CANDIDATES:
            _PREVIEW_CANDIDATES[op] = []

        _PREVIEW_CANDIDATES[op].append(cls)

    @classmethod
    def name(cls) -> str:
        return _camel_to_snake(cls.__name__)


def find_preview(
    op: _ops.OpOverload, *args: typing.Any, **kwargs: typing.Any
) -> Preview:
    """
    Try finding a preview with the given `op` and its arguments.
    """

    if op not in _PREVIEW_CANDIDATES:
        return NotImplemented

    for candidate in _PREVIEW_CANDIDATES[op]:
        if not (preview := candidate(*args, **kwargs)).ok():
            continue

        return preview

    return NotImplemented


def all_previews():
    return _PREVIEW_CANDIDATES


_CAMEL_CASE_REGEX = re.compile(r"(?<!^)(?=[A-Z])")


def _camel_to_snake(name: str) -> str:
    return re.sub(_CAMEL_CASE_REGEX, "_", name).lower()
