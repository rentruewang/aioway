# Copyright (c) AIoWay Authors - All Rights Reserved

"The utilities for signatures."

import collections
import copy
import dataclasses as dcls
import enum
import functools
import inspect
import typing
from collections import abc as cabc

import torch
from torch import nn, overrides
from torch.utils import _pytree as pyt

from aioway._utils import is_any_type_hint, is_nn_type, is_seq_of
from aioway.t.structs import TList

from .guards import is_torch_function
from .structs import PyTreeDict

__all__ = ["Param", "ParamKind", "Sign", "BoundArgs"]


class ParamKind(enum.Enum):
    "The type checked alternative to `inspect`'s private `.kind` type."

    POSITIONAL_ONLY = enum.auto()
    POSITIONAL_OR_KEYWORD = enum.auto()
    VAR_POSITIONAL = enum.auto()
    KEYWORD_ONLY = enum.auto()
    VAR_KEYWORD = enum.auto()

    def __eq__(self, other) -> bool:
        if isinstance(other, ParamKind):
            return self is other

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

    def __str__(self) -> str:
        return str(self.parameter)

    def __repr__(self) -> str:
        return repr(self.parameter)

    def __eq__(self, other) -> bool:
        if isinstance(other, Param):
            return self.parameter == other.parameter

        if isinstance(other, inspect.Parameter):
            return self.parameter == other

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
    def default_is_empty(self) -> bool:
        return self.default is inspect.Parameter.empty

    @property
    def annotation_is_empty(self) -> bool:
        return self.annotation is inspect.Parameter.empty

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


class Sign:
    "The convenient wrapper for `inspect.Signature`."

    def __init__(self, signature: inspect.Signature) -> None:
        self._sign = signature

    def __str__(self) -> str:
        return str(self._sign)

    def __repr__(self) -> str:
        return repr(self._sign)

    def __hash__(self) -> int:
        return hash(self._sign)

    def __eq__(self, other) -> bool:
        match other:
            case Sign():
                return self._sign == other._sign
            case inspect.Signature():
                return self._sign == other
            case _:
                return NotImplemented

    def args_annot(self) -> ArgsAnnot:
        "Get the arguments' annotations."

        return ArgsAnnot(
            {name: param.annotation for name, param in self.params.items()}
        )

    @property
    def params(self) -> dict[str, Param]:
        return {key: Param(param) for key, param in self._sign.parameters.items()}

    @functools.cached_property
    def param_list(self) -> tuple[Param, ...]:
        "Convert parameters into a list."
        return tuple(self.params.values())

    @property
    def return_annotation(self) -> type:
        return self._sign.return_annotation

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

    def rename(self, /, **names: str) -> typing.Self:
        """
        A copy with parameters renamed, like `dcls.replace`.
        """

        if unknown := names.keys() - self.params.keys():
            raise KeyError(f"Unknown parameters: {unknown}.")

        params = [
            p.parameter.replace(name=names.get(p.name, p.name)) for p in self.param_list
        ]
        return type(self)(self._sign.replace(parameters=params))

    def bind(self, *args, **kwargs) -> BoundArgs:
        """
        Bind the arguments and return a `BoundArguments`.

        Raises:
            TypeError: if binding failed.
        """

        return BoundArgs.from_signs(self, *args, **kwargs)

    def apply(self, *args, **kwargs) -> dict[str, typing.Any]:
        "Apply the signature on the *args, **kwargs and get the argument dict."

        return self.bind(*args, **kwargs).apply_defaults().arguments

    def drop_first(self) -> typing.Self:
        "Drop the `self` parameter."
        params = [param.parameter for param in self.param_list[1:]]
        return type(self)(self._sign.replace(parameters=params))

    @property
    def argc(self) -> int:
        return len(self._sign.parameters)

    def to_inspect(self) -> inspect.Signature:
        "Get the `inspect.Signature` representation."
        return self._sign

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
    def from_nn_forward_untyped(cls, module: type[nn.Module]) -> Sign:
        """
        Parse the `nn.Module.forward` without `self` or type.

        Caches the signature into `nn.Module` registry automatically.
        """

        return cls.from_nn_forward(module).strip_type()

    @classmethod
    def from_callable(cls, func: cabc.Callable) -> typing.Self:
        "Create the signature from a method."

        # Torch builtins are not inspectable, so we need dummies.
        if (
            True
            and is_torch_function(func)
            and func in (cpp_funcs := overrides.get_testing_overrides())
        ):
            func = cpp_funcs[func]

        if isinstance(func, nn.Module):
            return cls.from_nn_forward(func)

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


class BoundArgs(PyTreeDict, pytree_key=pyt.MappingKey):
    """
    Our version of `inspect.BoundArguments`.

    The `pyt` children are the bound arguments, keyed by parameter name.
    """

    __iter__ = None
    "Explicitly disallow iteration."

    def __init__(self, *, bounded: inspect.BoundArguments) -> None:
        self._bounded = bounded

    def __repr__(self) -> str:
        return repr(self._bounded)

    def __str__(self) -> str:
        return str(self._bounded)

    def __eq__(self, other) -> bool:
        if isinstance(other, BoundArgs):
            return self._bounded == other._bounded

        if isinstance(other, inspect.BoundArguments):
            return self._bounded == other

        if isinstance(other, dict):
            return self.arguments == other

        return NotImplemented

    def __hash__(self) -> int:
        return hash(self._bounded)

    def __copy__(self) -> typing.Self:
        return type(self)(bounded=self._bounded)

    def __len__(self) -> int:
        return len(self.arguments)

    def __getitem__(self, key: int | str) -> typing.Any:
        if isinstance(key, str):
            return self.arguments[key]

        if isinstance(key, int):
            params = self.sign.param_list
            return params[key]

        raise TypeError(f"Only handles ints and strs. Got {type(key)=}.")

    def keys(self) -> cabc.KeysView[str]:
        return self.arguments.keys()

    @property
    def sign(self) -> Sign:
        return Sign(signature=self._bounded.signature)

    @property
    def param_list(self) -> tuple[Param, ...]:
        return self.sign.param_list

    @functools.cached_property
    def arguments(self) -> dict[str, typing.Any]:
        """
        An ordered copy of the `BoundArguments.arguments` dictionary.
        """

        # Only bind the provided arguments, and in order.
        provided = self._bounded.arguments.keys()
        names_in_order = [p.name for p in self.param_list]
        return {n: self._bounded.arguments[n] for n in names_in_order if n in provided}

    @property
    def args(self) -> tuple:
        return self._bounded.args

    @property
    def kwargs(self) -> dict[str, typing.Any]:
        return self._bounded.kwargs

    @property
    def tensors(self) -> TList:
        return self._tensor_list

    @functools.cached_property
    def _tensor_list(self) -> TList:
        tensors = [
            tensor
            for tensor in pyt.tree_leaves(self.arguments)
            if isinstance(tensor, torch.Tensor)
        ]
        return TList.from_iterable(tensors)

    def apply_defaults(self) -> typing.Self:
        self = copy.copy(self)
        self._bounded.apply_defaults()
        return self

    def keys(self) -> cabc.KeysView[str]:
        return self.arguments.keys()

    def values(self) -> cabc.ValuesView[typing.Any]:
        return self.arguments.values()

    def items(self) -> cabc.ItemsView[str, typing.Any]:
        return self.arguments.items()

    def to_inspect(self) -> inspect.BoundArguments:
        return self._bounded

    @property
    @typing.override
    def _tree_context(self) -> tuple[Sign, tuple[str, ...]]:
        "The signature, and the names of the bound parameters."
        return self.sign, tuple(self.arguments)

    @typing.override
    def _tree_dict(self) -> dict[str, typing.Any]:
        return self.arguments

    @classmethod
    @typing.override
    def _tree_unflatten(cls, children: cabc.Iterable, context) -> typing.Self:
        values = list(children)
        signature, names = context

        assert isinstance(signature, Sign)
        assert len(names) == len(values)
        assert is_seq_of(str)(names), names

        arguments = {key: val for key, val in zip(names, values)}
        return cls(bounded=inspect.BoundArguments(signature.to_inspect(), arguments))

    @classmethod
    def from_signs(cls, sign: Sign, *args, **kwargs) -> typing.Self:
        bounded = sign.to_inspect().bind(*args, **kwargs)
        return cls(bounded=bounded)

    @classmethod
    def from_args(cls, sign: Sign, args: cabc.Mapping[str, typing.Any]) -> typing.Self:
        "Construct from a signature and the args by parameter name, without binding."

        if unknown := args.keys() - sign.params.keys():
            raise TypeError(f"Unknown parameters: {sorted(unknown)}.")

        # Signature order, so `args` / `kwargs` / iteration follow the signature.
        ordered = {name: args[name] for name in sign.params if name in args}
        return cls(bounded=inspect.BoundArguments(sign.to_inspect(), ordered))
