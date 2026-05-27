# Copyright (c) AIoWay Authors - All Rights Reserved

"Metadata for torch operators / functions."

import abc
import dataclasses as dcls
import functools
import logging
import typing
from collections import abc as cabc

import torch

from aioway._utils import find_nested_tensors, render_fcall

__all__ = ["Fn", "TensorInput", "Thunk", "TorchThunk", "thunk_dcls"]

LOGGER = logging.getLogger(__name__)
_PENDING = object()
"The object signifying a status of pending. This is a `object()` s.t. `FnCache` can store `None`."


@typing.runtime_checkable
class Fn[T = object](typing.Protocol):
    """
    `Fn` is the base class for delayed computation, in a single batch (a single pass).

    It stands for [f]unction [n]ode, or short for function.

    `Fn()` executes the computation, `Fn` base class itself does not make any more assumption.
    """

    def __call__(self) -> T:
        """
        Execute the computation.
        """

        raise NotImplementedError


@typing.runtime_checkable
class TensorInput(typing.Protocol):
    """
    `TensorInput` marks a class whose value depend on input tensors for computation.
    """

    def inputs(self) -> cabc.Iterable[torch.Tensor]:
        "The tensor operands (inputs to the function)"

        raise NotImplementedError


class Thunk:
    """
    The thunk for any function, handles both pretty printing and storing the result.

    `Thunk` has advantage over `functools.cache`, `functools.cached_property`,
    and having a saved `.__result` member for instance, by being the least assuming.

    `functools.cache` assumes that `self` is hashable.
    `functools.cached_property` cannot inspect whether we have evaluated it or not.
    `.__result` member assumes subclass calls `__init__` properly.

    Storing a `Thunk` in a `functools.cached_property` means `self` can be unhashable,
    the item can be inspected, and subclasses do not need to call `__init__`.

    Like Haskell's thunks, once evaluated,
    the value is stored in the `Thunk` itself and never re-evaluated.
    The value shall be gone during GC.

    I was going to go for `Op` but it's used a lot in `torch`.
    """

    def __init__(
        self,
        func: cabc.Callable[..., typing.Any],
        /,
        *args: typing.Any,
        **kwargs: typing.Any,
    ) -> None:
        if not callable(func):
            raise ValueError("Cannot create thunk for function that isn't callable.")

        self._func = func
        self._args = args
        self._kwargs = kwargs

    @typing.override
    def __eq__(self, other: object) -> bool:
        if isinstance(other, Thunk):
            return (
                True
                and self.func == other.func
                and self.args == other.args
                and self.kwargs == other.kwargs
            )

        return NotImplemented

    def __call__(self) -> object:
        return self.func(*self.args, **self.kwargs)

    @typing.override
    def __repr__(self) -> str:
        return self.__string

    @typing.override
    def __str__(self) -> str:
        return self.__string

    @property
    def thunk(self) -> Thunk:
        return self

    @property
    def func(self):
        return self._func

    @property
    def args(self):
        return self._args

    @property
    def kwargs(self):
        return self._kwargs

    @functools.cached_property
    def __string(self) -> str:
        args, kwargs = self.args, self.kwargs

        return render_fcall(self.func, *args, **kwargs)


@typing.dataclass_transform()
def thunk_dcls(cls: type):
    return dcls.dataclass(match_args=False)(cls)


@thunk_dcls
class TorchThunk[T: cabc.Callable[..., typing.Any]](abc.ABC):
    """
    `TorchThunk` is a really basic `Fn` that acts as a base class,
    with some `torch` utilities.
    """

    _: dcls.KW_ONLY

    func: T
    "The function to call. Must be callable.."

    args: tuple[typing.Any, ...]
    "The positional args."

    kwargs: dict[str, typing.Any]
    "The keyword arguments."

    def __post_init__(self):
        if not callable(self.func):
            raise TypeError(f"{self.func=} is not callable.")

        if not isinstance(self.args, tuple):
            raise TypeError(f"{self.args=} is not a tuple.")

        if not isinstance(self.kwargs, dict):
            raise TypeError(f"{self.kwargs=} is not a dict.")

    def __call__(self) -> object:
        return self.func(*self.args, **self.kwargs)

    def inputs(self):
        yield from find_nested_tensors(self.args)
        yield from find_nested_tensors(self.kwargs)

    @property
    def requires_grad(self) -> bool:
        "Check if any of the inputs requires grad."

        return any(tensor.requires_grad for tensor in self.inputs())
