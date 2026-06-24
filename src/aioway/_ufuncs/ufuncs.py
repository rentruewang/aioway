# Copyright (c) AIoWay Authors - All Rights Reserved

import abc
import functools
import inspect
import typing
from collections import abc as cabc

from aioway._iters import Iter, StructIter
from aioway._utils import decomp_flatten, decomp_replace, render_fcall

__all__ = ["UFunc", "UFuncThunk"]


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

        self._check_signature(*args, **kwargs)
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
        self._check_signature(*args, **kwargs)
        return self.THUNK(self, *args, **kwargs)

    @functools.cached_property
    def __signature__(self) -> inspect.Signature:
        "The signature of the `UFunc`."
        return inspect.signature(self.forward)

    def _check_signature(self, *args, **kwargs) -> None:
        # This only checks the signature names, not types,
        # so it's perfect for our usage, because `__call__`, `thunk`,
        # both share the same signature but with different types.

        try:
            _ = self.__signature__.bind(*args, **kwargs)
        except TypeError as te:
            raise TypeError(f"{self!r} gets a bad input.") from te
