# Copyright (c) AIoWay Authors - All Rights Reserved

import abc
import dataclasses as dcls
import typing
from collections import abc as cabc

import jinja2 as j2

from aioway._utils import AnyDict, Sign, decomp_flatten, decomp_replace

from .ufuncs import UFunc

__all__ = ["CompoundBuilder", "BuilderNode", "BuiltUFunc"]

TEMPLATE = j2.Template("""
class {{ module }}(nn.Module):
    def __init__(self, {{ init_signature }}):
        super().__init__()

        {% for stmt in init_stmts %}
        {{ stmt }}
        {% endfor %}

    def forward(self, {{ fwd_signature }}):
        {% for stmt in fwd_stmts %}
        {{ stmt }}
        {% endfor %}
""")


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

    def codegen(self, name: str) -> str:
        "Generate the definition."

        raise NotImplementedError


class BuilderNode(abc.ABC):
    """
    The thunk used by the builder, that will be evaluated
    """

    def __call__(self, node_vals: AnyDict[BuilderNode, typing.Any], /) -> typing.Any:
        "Calls `.compute` with memoization."

        if self not in node_vals:
            result = self.compute(node_vals)
            node_vals[self] = result

        return node_vals[self]

    @abc.abstractmethod
    def compute(self, node_vals: AnyDict[BuilderNode, typing.Any], /) -> typing.Any:
        """
        Evaluate the node values given the node values.
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
    def compute(self, node_vals: AnyDict[BuilderNode, typing.Any], /) -> typing.Any:
        # Compute handles cases wehre `self not in node_vals`, impossible for this type.
        raise KeyError(f"{self=} is not found in inputs.")

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
    def compute(self, node_vals: AnyDict[BuilderNode, typing.Any], /) -> typing.Any:
        compute_node = lambda node: (
            node(node_vals) if isinstance(node, BuilderNode) else NotImplemented
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
        all_kwargs = self._signature.apply(*args, **kwargs)

        if set(all_kwargs.keys()) != {i.name for i in self.inputs}:
            raise ValueError(
                "Not all the argumetns are supplied. "
                f"Found {all_kwargs}, but inputs should be {self.inputs}."
            )

        mapping_to_inputs = {input.name: input for input in self.inputs}
        node_vals = AnyDict[BuilderNode, typing.Any]()
        for key, val in all_kwargs.items():
            node_vals[mapping_to_inputs[key]] = val

        return self.output(node_vals)

    @property
    @typing.override
    def _signature(self) -> Sign:
        return Sign.from_inputs([input.name for input in self.inputs])
