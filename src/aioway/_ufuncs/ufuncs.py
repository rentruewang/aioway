# Copyright (c) AIoWay Authors - All Rights Reserved

import abc
import contextlib as ctxl
import dataclasses as dcls
import inspect
import typing
from collections import abc as cabc

from aioway._api import public_api
from aioway._iters import Iter, StructIter
from aioway._utils import Sign, decomp_flatten, decomp_replace, render_fcall

from .profs import UFuncProf

__all__ = ["UFunc", "UFuncThunk", "IdentityUFunc", "AdHocUFunc", "ufunc_profiler"]

_ufunc_profiler: UFuncProf = ctxl.nullcontext
"The profiler that would be entered."


class UFuncThunk[**P = ..., T = typing.Any](Iter[T]):
    "The `Thunk` / `Iter` adaptor type for `UFunc`."

    @typing.no_type_check
    def __init__(self, ufunc: UFunc[P, T], *args: P.args, **kwargs: P.kwargs):
        self._ufunc: UFunc[P, T] = ufunc
        self._args: typing.Any = args
        self._kwargs: typing.Any = kwargs

    def __repr__(self):
        return render_fcall(self.ufunc, *self.args, **self.kwargs)

    def __call__(self) -> T:
        def eval_thunk(thunk):
            """
            If `UFuncThunk`, evaluate. Else do nothing.
            """

            if not isinstance(thunk, UFuncThunk):
                return NotImplemented

            return thunk()

        args = decomp_replace(self.args, eval_thunk)
        kwargs = decomp_replace(self.kwargs, eval_thunk)

        return self.ufunc(*args, **kwargs)

    def iterate(self) -> cabc.Generator[T]:
        args_iter = StructIter(self.args)
        kwargs_iter = StructIter(self.kwargs)

        for args, kwargs in zip(args_iter, kwargs_iter):
            assert isinstance(args, cabc.Sequence), args
            assert isinstance(kwargs, cabc.Mapping), kwargs
            yield self.ufunc(*args, **kwargs)

    @property
    def ufunc(self) -> UFunc[P, T]:
        return self._ufunc

    @property
    def args(self):
        return self._args

    @property
    def kwargs(self):
        return self._kwargs

    def deps(self) -> cabc.Iterator[UFuncThunk]:
        "Get all the input thunks."
        yield from decomp_flatten(self.args, UFuncThunk)
        yield from decomp_flatten(self.kwargs, UFuncThunk)


@public_api
class UFunc[**P, T](abc.ABC):
    """
    `UFunc`, inspired by `numpy`, stands for universal functions,
    and are the building blocks of other APIs (like thunks and iters).
    """

    THUNK: typing.ClassVar[type[UFuncThunk]] = UFuncThunk
    "The thunk class used in the `.thunk` function."

    def __str__(self) -> str:
        "Repr is equivalent to object + signature."
        return object.__repr__(self) + str(self.__signature__)

    def __call__(self, *args, **kwargs) -> typing.Any:
        """
        A `UFunc` is callable (imperative).
        The calling logic is defined in `forward`.

        The signature of it would be checked prior to calling.
        """

        self.validate_signature(*args, **kwargs)

        with _ufunc_profiler(self):
            return self.forward(*args, **kwargs)

    def _repr(self) -> str:
        "The `__repr__` of the instance itself, excluding signature."
        return object.__repr__(self)

    @abc.abstractmethod
    @typing.no_type_check
    def forward(self, *args: P.args, **kwargs: P.kwargs) -> T:
        raise NotImplementedError

    def thunk(self, *args, **kwargs) -> typing.Any:
        "A `UFunc` takes thunks and tranform it into other thunks."
        self.validate_signature(*args, **kwargs)
        return self.THUNK(self, *args, **kwargs)

    @property
    def __signature__(self) -> inspect.Signature:
        return self._signature.signature

    @property
    def _signature(self) -> Sign:
        "The signature of the `UFunc`."
        return Sign.from_callable(self.forward)

    def validate_signature(self, *args, **kwargs) -> None:
        # This only checks the signature names, not types,
        # so it's perfect for our usage, because `__call__`, `thunk`,
        # both share the same signature but with different types.

        try:
            _ = self._signature.bind(*args, **kwargs)
        except TypeError as te:
            raise TypeError(f"{self!r} gets a bad input.") from te


@ctxl.contextmanager
def ufunc_profiler(prof: UFuncProf) -> cabc.Generator[None]:
    "Set the global profiler to the custom one."

    global _ufunc_profiler
    before, _ufunc_profiler = _ufunc_profiler, prof

    try:
        yield
    finally:
        _ufunc_profiler = before


@public_api
@dcls.dataclass(frozen=True)
class IdentityUFunc[T](UFunc):
    """
    A `UFunc` that passes the input forward to the output.
    """

    def forward(self, arg: T) -> T:
        return arg


@public_api
@dcls.dataclass(frozen=True)
class AdHocUFunc[**P, T](UFunc):
    """
    An ad-hoc `UFunc` that is useful for wrapping other functions.
    """

    function: cabc.Callable[P, T]
    "The function to wrap."

    @typing.override
    def forward(self, *args: P.args, **kwargs: P.kwargs) -> T:
        return self.function(*args, **kwargs)

    @property
    @typing.override
    def _signature(self):
        return Sign.from_callable(self.function)
