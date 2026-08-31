# Copyright (c) AIoWay Authors - All Rights Reserved

"The utilities for signatures."

import dataclasses as dcls
import inspect
import typing
from collections import abc as cabc

__all__ = ["Param", "Sign"]


@dcls.dataclass(frozen=True)
class Param:
    "The convenient wrapper for `inspect.Parameter`."

    parameter: inspect.Parameter
    "The underlying parameter."

    def __repr__(self) -> str:
        return repr(self.parameter)

    def __eq__(self, other) -> bool:
        match other:
            case Param(parameter=param):
                return self.parameter == param
            case inspect.Parameter():
                return self.parameter == other
            case _:
                return NotImplemented

    @property
    def name(self) -> str:
        return self.parameter.name

    @property
    def kind(self):
        return self.parameter.kind

    @property
    def default(self):
        return self.parameter.default

    @property
    def annotation(self) -> type:
        return self.parameter.annotation

    @property
    def is_any_type(self) -> bool:
        return self.annotation is inspect.Parameter.empty

    def strip_type(self) -> typing.Self:
        "Strip the type from `inspect.Parameter`."

        return type(self)(
            inspect.Parameter(
                name=self.name,
                kind=self.kind,
                default=self.default,
                annotation=inspect.Parameter.empty,
            )
        )


@dcls.dataclass(frozen=True)
class Sign:
    "The convenient wrapper for `inspect.Signature`."

    signature: inspect.Signature
    "The underlying signature."

    def __repr__(self) -> str:
        return repr(self.signature)

    def __eq__(self, other) -> bool:
        match other:
            case Sign(signature=sign):
                return self.signature == sign
            case inspect.Signature():
                return self.signature == other
            case _:
                return NotImplemented

    @property
    def params(self) -> dict[str, Param]:
        return {key: Param(param) for key, param in self.signature.parameters.items()}

    @property
    def param_list(self) -> list[Param]:
        "Convert parameters into a list."
        return list(self.params.values())

    @property
    def return_annotation(self) -> type:
        return self.signature.return_annotation

    @property
    def returns_any_type(self) -> bool:
        "Whether or not the return type is defined."
        return self.return_annotation is inspect.Parameter.empty

    def strip_type(self) -> typing.Self:
        """
        The outline of the signature discards all typing information.
        """

        params: list[inspect.Parameter] = [
            param.strip_type().parameter for param in self.params.values()
        ]

        return type(self)(inspect.Signature(parameters=params))

    def bind(self, *args, **kwargs) -> inspect.BoundArguments:
        """
        Bind the arguments and return a `BoundArguments`.

        Raises:
            TypeError: if binding failed.
        """

        return self.signature.bind(*args, **kwargs)

    def apply(self, *args, **kwargs) -> dict[str, typing.Any]:
        "Apply the signature on the *args, **kwargs and get the argument dict."

        bound = self.bind(*args, **kwargs)
        bound.apply_defaults()
        arguments = bound.arguments
        assert isinstance(arguments, dict), arguments
        return arguments

    def drop_first(self) -> typing.Self:
        "Drop the `self` parameter."
        params = [param.parameter for param in self.param_list[1:]]
        return type(self)(inspect.Signature(params))

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
