# Copyright (c) AIoWay Authors - All Rights Reserved

import abc
import collections
import dataclasses as dcls
import typing
from collections import abc as cabc

import jinja2 as j2
from torch import nn

from aioway._utils import (
    AnyDict,
    Sign,
    decomp_flatten,
    decomp_replace,
    render_fcall,
)

from .ufuncs import UFunc

__all__ = ["CompoundBuilder", "BuilderNode", "BuiltUFunc"]

_CODEGEN_TEMPLATE: j2.Template = j2.Template("""
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

    def __init__(self) -> None:
        self.nodes: list[BuilderNode] = []
        "The nodes corresponding to the variables."

        self._ufunc_names: AnyDict[UFunc, str] = AnyDict()
        "The names of the ufuncs currently in scope."

        self._type_count: dict[type[UFunc], int] = collections.defaultdict(int)
        "Mapping from ufunc types to the number of occurences."

    def input(self, name: str) -> InputBuilderNode:
        "Set the input nodes. Translate to the argument list."

        # Since the nodes are referred to by `id`, we need to find it exactly.
        for input in self.inputs():
            if input.name == name:
                return input

        node = InputBuilderNode(name)
        self.nodes.append(node)
        return node

    def thunk(self, ufunc: UFunc, *args, **kwargs) -> ThunkBuilderNode:
        "The thunks will translate to the statements."

        node = ThunkBuilderNode(ufunc, *args, **kwargs)
        self.nodes.append(node)

        # If the name does not exist, assign a name.
        if ufunc not in self._ufunc_names:
            self._ufunc_names[ufunc] = self._ufunc_new_name(ufunc)

        return node

    def inputs(self) -> list[InputBuilderNode]:
        "Get all the inputs registered."
        return [node for node in self.nodes if isinstance(node, InputBuilderNode)]

    def thunks(self) -> list[ThunkBuilderNode]:
        "Get all the thunks."
        return [node for node in self.nodes if isinstance(node, ThunkBuilderNode)]

    def output(self, node: BuilderNode) -> BuiltUFunc:
        "Set the output and returns a `UFunc`."
        return BuiltUFunc(builder=self, output=node)

    def _ufunc_new_name(self, ufunc: UFunc) -> str:
        # Count the number of the same type, to add suffix.
        ufunc_type = type(ufunc)
        self._type_count[ufunc_type] += 1
        count = self._type_count[ufunc_type]

        return f"{ufunc_type.__name__}_{count}"


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

    def __repr__(self):
        return render_fcall(self.ufunc, *self.args, **self.kwargs)

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

    builder: CompoundBuilder
    "The builder itself."

    output: BuilderNode
    "The output to evaluate."

    def forward(self, *args, **kwargs):
        all_kwargs = self._signature.apply(*args, **kwargs)

        if set(all_kwargs.keys()) != {i.name for i in self.inputs}:
            raise ValueError(
                "Not all the argumetns are supplied. "
                f"Found {all_kwargs}, but inputs should be {self.inputs}."
            )

        mapping_to_inputs = self._input_names
        node_vals = AnyDict[BuilderNode, typing.Any]()
        for key, val in all_kwargs.items():
            node_vals[mapping_to_inputs[key]] = val

        return self.output(node_vals)

    @property
    @typing.override
    def _signature(self) -> Sign:
        return Sign.from_inputs([input.name for input in self.inputs])

    @property
    def inputs(self) -> list[InputBuilderNode]:
        return self.builder.inputs()

    @property
    def _input_names(self) -> dict[str, InputBuilderNode]:
        return {input.name: input for input in self.inputs}

    def _names_of(self, ufunc: UFunc) -> str:
        return self.builder._ufunc_names[ufunc]

    def parameters(self) -> cabc.Generator[nn.Parameter]:
        from aioway.torch.nn_ import NnUFunc

        for node in self.builder.nodes:
            if isinstance(node, ThunkBuilderNode) and isinstance(node.ufunc, NnUFunc):
                yield from node.ufunc.module.parameters()

    def codegen(self, name: str) -> str:
        "Generate the definition."

        def render_builder_node(node):
            if isinstance(node, InputBuilderNode):
                return node.name

            if isinstance(node, ThunkBuilderNode):
                return self._names_of(node.ufunc)

            return NotImplemented

        def render_fwd_stmt_rhs(thunk: ThunkBuilderNode):

            args = decomp_replace(thunk.args, render_builder_node)
            kwargs = decomp_replace(thunk.kwargs, render_builder_node)
            return render_fcall(f"self.{self._names_of(thunk.ufunc)}", *args, **kwargs)

        return _CODEGEN_TEMPLATE.render(
            module=name,
            init_signature=", ".join(self.builder._ufunc_names.values()),
            init_stmts=[
                f"self.{name} = {name}" for name in self.builder._ufunc_names.values()
            ],
            fwd_signature=", ".join(self._input_names.keys()),
            fwd_stmts=[
                f"{self._names_of(thunk.ufunc)} = {render_fwd_stmt_rhs(thunk)}"
                for thunk in self.builder.thunks()
            ],
        )
