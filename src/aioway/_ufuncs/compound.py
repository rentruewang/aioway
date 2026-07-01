# Copyright (c) AIoWay Authors - All Rights Reserved

import abc
import dataclasses as dcls
import inspect
import typing
from collections import abc as cabc

from aioway._utils import AnyDict, decomp_flatten, decomp_replace

from .ufuncs import UFunc

__all__ = ["CompoundBuilder", "BuilderNode", "BuiltUFunc"]


@dcls.dataclass
class CompoundBuilder:
    """
    The builder for the compounds.
    Stores all the thunks sequentially, to convert to statements.
    """

    nodes: list[BuilderNode] = dcls.field(default_factory=list)
    "The nodes corresponding to the variables."

    ufunc_names: AnyDict[UFunc, str] = dcls.field(default_factory=AnyDict)
    "The names of the ufuncs currently in scope."

    type_count: AnyDict[type[UFunc], int] = dcls.field(default_factory=AnyDict)
    "Mapping from ufunc types to the number of occurences."

    def input(self, name: str) -> InputBuilderNode:
        "Set the input nodes. Translate to the argument list."

        node = InputBuilderNode(name)
        self.nodes.append(node)
        return node

    def thunk(self, ufunc: UFunc, *args, **kwargs) -> ThunkBuilderNode:
        "The thunks will translate to the statements."

        node = ThunkBuilderNode(ufunc, *args, **kwargs)
        self.nodes.append(node)

        # If the name does not exist, assign a name.
        if ufunc not in self.ufunc_names:
            self.ufunc_names[ufunc] = self._ufunc_new_name(ufunc)

        return node

    def inputs(self) -> list[InputBuilderNode]:
        "Get all the inputs registered."
        return [node for node in self.nodes if isinstance(node, InputBuilderNode)]

    def output(self, node: BuilderNode) -> BuiltUFunc:
        "Set the output and returns a `UFunc`."
        return BuiltUFunc(inputs=self.inputs(), output=node)

    def _ufunc_new_name(self, ufunc: UFunc) -> str:
        # Count the number of the same type, to add suffix.
        ufunc_type = type(ufunc)
        self.type_count[ufunc_type] = self.type_count.get(ufunc_type, 0) + 1
        count = self.type_count[ufunc_type]

        return f"{ufunc_type.__name__}_{count}"


class BuilderNode(abc.ABC):
    """
    The thunk used by the builder, that will be evaluated
    """

    @abc.abstractmethod
    def compute(self, inputs: dict[str, typing.Any], /) -> typing.Any:
        """
        Evaluate the node values given the inputs.
        """

        raise NotImplementedError

    @abc.abstractmethod
    def deps(self) -> cabc.Iterator[BuilderNode]:
        raise NotImplementedError


@dcls.dataclass
class InputBuilderNode(BuilderNode):
    """
    The class that signals that there is an input in this place.
    """

    name: str
    """
    The name of the input.
    """

    @typing.override
    def compute(self, inputs: dict[str, typing.Any]) -> typing.Any:
        if self.name not in inputs:
            raise KeyError(f"{self.name=} is not supplied in {inputs=}.")

        return inputs[self.name]

    def deps(self):
        return
        yield


@dcls.dataclass
class ThunkBuilderNode(BuilderNode):
    """
    The class that holds a `UFunc`.
    """

    ufunc: UFunc
    args: tuple[typing.Any, ...]
    kwargs: dict[str, typing.Any]

    def __init__(self, ufunc: UFunc, *args, **kwargs):
        ufunc.validate_signature(*args, **kwargs)

        self.ufunc = ufunc
        self.args = args
        self.kwargs = kwargs

    @typing.override
    def compute(self, inputs: dict[str, typing.Any]) -> typing.Any:
        compute_node = lambda node: (
            node.compute(inputs) if isinstance(node, BuilderNode) else NotImplemented
        )

        args = decomp_replace(self.args, compute_node)
        kwargs = decomp_replace(self.kwargs, compute_node)

        return self.ufunc(*args, **kwargs)

    def deps(self):
        yield from decomp_flatten(self.args, BuilderNode)
        yield from decomp_flatten(self.kwargs, BuilderNode)


@dcls.dataclass
class BuiltUFunc(UFunc):
    "The ufunc that traces from outputs to inputs with a pull strategy."

    inputs: list[InputBuilderNode]
    "The input nodes. `.compute`'s dict's key must correspond to the names here."

    output: BuilderNode
    "The output to evaluate."

    def forward(self, *args, **kwargs):
        all_kwargs = apply_signature(self.__signature__, *args, **kwargs)
        return self.output.compute(all_kwargs)

    @property
    @typing.override
    def __signature__(self):
        return inspect.Signature(
            [
                inspect.Parameter(
                    input.name, kind=inspect.Parameter.POSITIONAL_OR_KEYWORD
                )
                for input in self.inputs
            ]
        )


def apply_signature(
    signature: inspect.Signature, *args, **kwargs
) -> dict[str, typing.Any]:
    bound = signature.bind(*args, **kwargs)
    bound.apply_defaults()
    all_kwargs = bound.arguments
    assert isinstance(all_kwargs, dict), all_kwargs
    return all_kwargs
