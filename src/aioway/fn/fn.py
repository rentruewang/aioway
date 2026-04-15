# Copyright (c) AIoWay Authors - All Rights Reserved

import abc
import functools
import typing
from collections import abc as cabc

import torch
from torch import _ops

from aioway.ctx import enabled_fake_mode, fake_mode_func, is_fake_tensor

__all__ = ["DoFn", "Fn", "TorchFn"]

_PENDING = object()
"The object signifying a status of pending. This is a `object()` s.t. `FnCache` can store `None`."


class Fn[**P, T]:
    """
    The thunk for any function, handles both pretty printing and storing the result.

    `Fn` has advantage over `functools.cache`, `functools.cached_property`,
    and having a saved `.__result` member for instance, by being the least assuming.

    `functools.cache` assumes that `self` is hashable.
    `functools.cached_property` cannot inspect whether we have evaluated it or not.
    `.__result` member assumes subclass calls `__init__` properly.

    Storing a `Fn` in a `functools.cached_property` means `self` can be unhashable,
    the item can be inspected, and subclasses do not need to call `__init__`.

    I was going to go for `Op` but it's used a lot in `torch`.
    """

    def __init__(
        self, func: cabc.Callable[P, T], /, *args: P.args, **kwargs: P.kwargs
    ) -> None:
        if not callable(func):
            raise ValueError("Cannot create thunk for function that isn't callable.")

        self._func = func
        self._args = args
        self._kwargs = kwargs

        self.__result: object = _PENDING
        "If not evaluated, it's pending."

    @typing.override
    def __eq__(self, other: object) -> bool:
        if isinstance(other, Fn):
            return (
                True
                and self.func == other.func
                and self.args == other.args
                and self.kwargs == other.kwargs
            )

        return NotImplemented

    def do(self) -> T:
        """
        Do the computation for `Fn`. Can be overwritten in subclass.
        """

        return self.func(*self.args, **self.kwargs)

    def __call__(self) -> T:
        """
        Call and cache the function.
        """

        if self.__result is _PENDING:
            self.__result = self.do()

        return typing.cast(T, self.__result)

    @typing.override
    def __repr__(self) -> str:
        return self.__string

    @typing.override
    def __str__(self) -> str:
        return self.__string

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
        pretty = _format_function_as_str(self.func, *self.args, **self.kwargs)

        if self.done:
            pretty = f"{pretty} -> {self.__result!r}"

        return pretty

    @property
    def done(self) -> bool:
        "Returns if the cache is previously called."

        return self.__result is not _PENDING


class TorchFn[**P, T](Fn[P, T]):
    def __init__(
        self,
        func: cabc.Callable[P, T],
        types: tuple[type, ...],
        /,
        *args: P.args,
        **kwargs: P.kwargs,
    ) -> None:
        super().__init__(func, *args, **kwargs)
        self._types = types

    @typing.override
    def __eq__(self, other: object) -> bool:
        if isinstance(other, TorchFn):
            return super().__eq__(other) and self.types == other.types

        return NotImplemented

    @property
    @typing.override
    @typing.no_type_check
    def func(self) -> _ops.OpOverload:
        return self._func

    @property
    def types(self):
        return self._types

    def parameters(self):
        def raw_params():
            yield from self.args
            yield from self.kwargs.values()

        for arg in raw_params():
            if isinstance(arg, torch.Tensor):
                yield arg


class PatchTorchFn[**P, T](TorchFn[P, T]):
    def __init__(
        self,
        func: cabc.Callable[P, T],
        patch: cabc.Callable[P, T],
        types: tuple[type, ...],
        /,
        *args: P.args,
        **kwargs: P.kwargs,
    ) -> None:
        super().__init__(func, types, *args, **kwargs)
        self._patch = patch

    @typing.override
    def do(self) -> T:
        if (patched := self.patch(*self.args, **self.kwargs)) is not NotImplemented:
            return patched

        return self.func(*self.args, **self.kwargs)

    @property
    def patch(self):
        return self._patch


class TorchIrFakePatchFn[**P, T](TorchFn):
    def __init__(
        self,
        func: _ops.OpOverload,
        types: tuple[type, ...],
        /,
        *args: P.args,
        **kwargs: P.kwargs,
    ) -> None:
        super().__init__(func, types, *args, **kwargs)


def _format_function_as_str(
    func: cabc.Callable[..., typing.Any], *args: typing.Any, **kwargs: typing.Any
) -> str:
    args_builder: list[str] = []

    # Add positional arguments.
    if args:
        args_builder.extend(f"{arg!r}" for arg in args)

    # Add keyword arguments.
    if kwargs:
        args_builder.extend(f"{k!s}={v!r}" for k, v in kwargs.items())

    args_str = ", ".join(args_builder)
    func_str = func.__qualname__
    return f"{func_str}({args_str})"


class DoFn[T](abc.ABC):
    """
    `DoFn`s represent computation that shall be done later.
    Right now, `DoFn` acts as an lazy version / augmentation of fake mode,
    patching some unsupported operations with worst case scenario (e.g. bool masking).

    Like Haskell's thunks, once evaluated,
    the value is stored in the `Fn` itself and never re-evaluated.
    The value shall be gone during GC.
    """

    __match_args__: typing.ClassVar[tuple[str, ...]]

    def __repr__(self):
        return self._name()

    @typing.final
    def do(self) -> T:
        """
        Perform the computation that is represented by this `Fn`.

        This is the public function that parent `Fn`s should call,
        when the want to request the values of an `Fn`.

        It handles caching in the normal case, so repeated calling the function means that
        the expensive computation (defined within `forward`) would only be called once.

        When fake mode is enabled, it calls `preview` for a fake tensor,
        which is a preview for the normal computation to save computation cost.

        The reason this is modal with `fake.is_enabled()` as a toggle,
        to make sure `preview` and `forward` can use the same codepath as much as possible,
        in the default case `preview` is `forward` with fake mode on.
        """

        if enabled_fake_mode():
            return self.preview()

        else:
            return self.__forward_cache()

    @fake_mode_func
    def preview(self) -> T:
        """
        The `preview` function generates a "preview" for the `Tensor` that would be generated.
        Should recursively call the dependent `Fn.do` functions.

        The result type (`FakeTensor`) is used as a worst case analysis of the original `Tensor`.

        In most cases (non leaf operators), this method is just a clone of `forward`,
        which is the default implementation of this function.

        In the following cases it must be modified:

        1. Source tensors, `forward` won't be `FakeTensor`, so conversion is needed.
        2. Operators that cannot be supported by `torch` e.g. boolean  masking.
        """

        result = self.forward()
        assert is_fake_tensor(result)
        return result

    @abc.abstractmethod
    def forward(self) -> T:
        raise NotImplementedError

    def deps(self) -> cabc.Generator[DoFn[typing.Any]]:
        """
        The `Fn`s that must be evaluated before we can evaluate the current `Fn`.

        Calling `do` on the current `Fn` would recursively call those.
        """

        # Inspect the fields of the `Fn`.
        # If sub-`Fn`s are found, also yield from those.
        for obj in self.__dict__.values():
            if isinstance(obj, DoFn):
                yield obj
                yield from obj.deps()

    @typing.final
    def parameters(self, deps: bool = True) -> cabc.Generator[torch.Tensor]:
        """
        Yield all the dependent parameters of `self`.

        Args:
            deps: If `True`, also yield the parameters from the dependent `Fn`s.

        Yields:
            The dependent tensors that are sources.
            Tensors that will be fake in fake mode.
        """

        yield from self._params_self()

        if not deps:
            return

        # Parameter `deps` is `True`, recursively get the data.
        for dep in self.deps():
            yield from dep.parameters(True)

    def _params_self(self) -> cabc.Generator[torch.Tensor]:
        """
        Yield parameters of `self`.

        The default implementation yields nothing,
        so subclasses should overwrite it if it is a source.
        """

        return
        yield

    @functools.cached_property
    def __forward_cache(self):
        return Fn(self.forward)

    @property
    def done(self) -> bool:
        "Whether or not this is done."

        return self.__forward_cache.done

    def _name(self) -> str:
        """
        The name of the `Fn` used in `repr`.
        """

        return type(self).__name__
