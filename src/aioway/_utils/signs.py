# Copyright (c) AIoWay Authors - All Rights Reserved

"The utilities for signatures."

import enum
import collections
import dataclasses as dcls
import inspect
import typing
from collections import abc as cabc

from torch import nn

from .typing import is_any_type_hint, is_nn_type

__all__ = ["Param", "ParamKind", "Sign"]


class ParamKind(enum.Enum):
    "The type checked alternative to `inspect`'s private `.kind` type."

    POSITIONAL_ONLY = enum.auto()
    POSITIONAL_OR_KEYWORD = enum.auto()
    VAR_POSITIONAL = enum.auto()
    KEYWORD_ONLY = enum.auto()
    VAR_KEYWORD = enum.auto()

    def __eq__(self, other) -> bool:
        # Use the typical enum implementation.
        if (result := super().__eq__(other)) is not NotImplemented:
            return result

        # Works with comparing with other `inspect.Parameter` kinds.
        if type(kind := self.to_inspect()) == type(other):
            return kind == other

        # Works with strings.
        if isinstance(other, str):
            return self.name == other.upper()

        return NotImplemented

    def __hash__(self):
        return super().__hash__()

    def to_inspect(self):
        "The matching `inspect.Parameter` kind."

        return getattr(inspect.Parameter, self.name)

    @classmethod
    def from_inspect(cls, kind) -> typing.Self:
        "Convert an `inspect.Parameter` kind to `ParamKind`."

        return cls[kind.name]


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
    def kind(self) -> ParamKind:
        return ParamKind.from_inspect(self.parameter.kind)

    @property
    def default(self):
        return self.parameter.default

    @property
    def annotation(self) -> type:
        return self.parameter.annotation

    @property
    def is_positional_only(self) -> bool:
        return self.kind == ParamKind.POSITIONAL_ONLY

    @property
    def is_keyword_only(self) -> bool:
        return self.kind == ParamKind.KEYWORD_ONLY

    @property
    def is_positional_or_keyword(self) -> bool:
        return self.kind == ParamKind.POSITIONAL_OR_KEYWORD

    @property
    def is_var_positional(self) -> bool:
        return self.kind == ParamKind.VAR_POSITIONAL

    @property
    def is_var_keyword(self) -> bool:
        return self.kind == ParamKind.VAR_KEYWORD

    @property
    def is_any_type(self) -> bool:
        return is_any_type_hint(self.annotation)

    def strip_type(self) -> typing.Self:
        "Strip the type from `inspect.Parameter`."

        return type(self)(
            inspect.Parameter(
                name=self.name,
                kind=self.kind.to_inspect(),
                default=self.default,
                annotation=inspect.Parameter.empty,
            )
        )

    @classmethod
    def build(
        cls,
        name: str,
        kind: ParamKind = ParamKind.POSITIONAL_OR_KEYWORD,
        *,
        default: typing.Any = inspect.Parameter.empty,
        annotation: typing.Any = inspect.Parameter.empty,
    ) -> typing.Self:
        "Construct a `Param` directly, mirroring `inspect.Parameter`'s constructor."

        return cls(
            inspect.Parameter(
                name=name,
                kind=kind.to_inspect(),
                default=default,
                annotation=annotation,
            )
        )


class ArgsAnnot(collections.UserDict[str, type]):
    "The argument annotation should be a mapping of name to type."

    def __eq__(self, other) -> bool:
        if not isinstance(other, ArgsAnnot):
            return NotImplemented

        # Keys should be equal.
        if self.keys() != other.keys():
            return False

        for key in self.keys():
            self_type = self[key]
            other_type = other[key]

            # If either is `typing.Any` or missing or `object`, skip.
            if is_any_type_hint(self_type) or is_any_type_hint(other_type):
                continue

            if self_type != other_type:
                return False

        return True


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

    def args_annot(self) -> ArgsAnnot:
        "Get the arguments' annotations."

        return ArgsAnnot(
            {name: param.annotation for name, param in self.params.items()}
        )

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
        return is_any_type_hint(self.return_annotation)

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
        return type(self)(self.signature.replace(parameters=params))

    @property
    def argc(self) -> int:
        return len(self.signature.parameters)

    @classmethod
    def from_nn_forward(cls, module: nn.Module | type[nn.Module]) -> typing.Self:
        """
        Get the signature of `.forward` function, excluding `self`,
        from type or instance of `nn.Module`.
        """

        if isinstance(module, nn.Module):
            module = type(module)

        if not is_nn_type(module):
            raise TypeError(f"{module=} should be a subclass of `nn.Module`.")

        return cls.from_callable(module.forward).drop_first()

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
