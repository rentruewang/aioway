# Copyright (c) AIoWay Authors - All Rights Reserved

"The utilities for signatures."

import dataclasses as dcls
import inspect
import typing
from collections import abc as cabc

__all__ = ["Sign"]


@dcls.dataclass(frozen=True)
class Sign:
    "The convenient wrapper for signature."

    signature: inspect.Signature
    "The underlying signature."

    def __repr__(self) -> str:
        return repr(self.signature)

    def __eq__(self, other) -> bool:
        if isinstance(other, Sign):
            return self.signature == other.signature

        if isinstance(other, inspect.Signature):
            return self.signature == other

        return NotImplemented

    @property
    def parameters(self) -> cabc.Mapping[str, inspect.Parameter]:
        return self.signature.parameters

    @property
    def return_annotation(self) -> typing.Any:
        return self.signature.return_annotation

    def bind(self, *args, **kwargs) -> inspect.BoundArguments:
        """
        Bind the arguments and return a `BoundArguments`. Raise `TypeError` if binding failed.
        """

        return self.signature.bind(*args, **kwargs)

    def apply(self, *args, **kwargs) -> dict[str, typing.Any]:
        "Apply the signature on the *args, **kwargs and get the argument dict."

        bound = self.bind(*args, **kwargs)
        bound.apply_defaults()
        arguments = bound.arguments
        assert isinstance(arguments, dict), arguments
        return arguments

    @property
    def argc(self) -> int:
        return len(self.signature.parameters)

    @classmethod
    def from_callable(cls, func: cabc.Callable) -> typing.Self:
        "Create the signature from a method."
        return cls(inspect.signature(func))

    @classmethod
    def from_inputs(cls, names: cabc.Iterable[str]) -> typing.Self:
        "Create the signature from input parameters."
        return cls(
            inspect.Signature(
                parameters=[
                    inspect.Parameter(
                        name, kind=inspect.Parameter.POSITIONAL_OR_KEYWORD
                    )
                    for name in names
                ]
            )
        )
