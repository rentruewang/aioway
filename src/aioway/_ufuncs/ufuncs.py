# Copyright (c) AIoWay Authors - All Rights Reserved

import abc
import inspect
import typing

__all__ = ["UFunc"]


class UFunc(abc.ABC):
    """
    `UFunc`, inspired by `numpy`, stands for universal functions,
    and are the building blocks of other APIs (like thunks and iters).
    """

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
    def forward(self, *args, **kwargs) -> T:
        raise NotImplementedError

    def thunk(self, *args, **kwargs) -> typing.Any:
        "A `UFunc` takes thunks and tranform it into other thunks."
        self._check_signature(*args, **kwargs)
        raise NotImplementedError

    def iter(self, *args, **kwargs) -> typing.Any:
        "A `UFunc` takes iterators and tranform it into other iterators."
        self._check_signature(*args, **kwargs)
        raise NotImplementedError

    @property
    def __signature__(self) -> inspect.Signature:
        "The signature of the `UFunc`."
        return inspect.signature(self.forward)

    def _check_signature(self, *args, **kwargs) -> None:
        # This only checks the signature names, not types,
        # so it's perfect for our usage, because `__call__`, `thunk`, `iter`
        # all share the same signature but with different types.

        try:
            _ = self.__signature__.bind(*args, **kwargs)
        except TypeError as te:
            raise TypeError(f"{self!r} gets a bad input.") from te
