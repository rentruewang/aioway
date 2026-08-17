# Copyright (c) AIoWay Authors - All Rights Reserved

import abc
import collections
import dataclasses as dcls
import typing
from collections import abc as cabc

from torch import nn

from aioway._utils import AnyDict, Sign, decomp_flatten, decomp_replace, render_fcall

__all__ = ["CompoundBuilder", "BuilderNode", "BuiltModule"]


@dcls.dataclass
class CompoundBuilder:
    """
    The builder for the compounds.
    Stores all the thunks sequentially, to convert to statements.
    """

    def __init__(self) -> None:
        self.nodes: list[BuilderNode] = []
        "The nodes corresponding to the variables."

        self._module_names: AnyDict[nn.Module, str] = AnyDict()
        "The names of the modules currently in scope."

        self._type_count: dict[type[nn.Module], int] = collections.defaultdict(int)
        "Mapping from module types to the number of occurences."

    def input(self, name: str) -> InputBuilderNode:
        "Set the input nodes. Translate to the argument list."

        # Since the nodes are referred to by `id`, we need to find it exactly.
        for input in self.inputs():
            if input.name == name:
                return input

        node = InputBuilderNode(name)
        self.nodes.append(node)
        return node

    def thunk(self, module: nn.Module, *args, **kwargs) -> ThunkBuilderNode:
        "The thunks will translate to the statements."

        node = ThunkBuilderNode(module, *args, **kwargs)
        self.nodes.append(node)

        # If the name does not exist, assign a name.
        if module not in self._module_names:
            self._module_names[module] = self._new_name_for_module(module)

        return node

    def inputs(self) -> list[InputBuilderNode]:
        "Get all the inputs registered."
        return [node for node in self.nodes if isinstance(node, InputBuilderNode)]

    def thunks(self) -> list[ThunkBuilderNode]:
        "Get all the thunks."
        return [node for node in self.nodes if isinstance(node, ThunkBuilderNode)]

    def output(self, node: BuilderNode) -> BuiltModule:
        "Set the output and returns an `nn.Module`."
        return BuiltModule(builder=self, output=node)

    def modules(self) -> cabc.Generator[nn.Module]:
        yield from self._module_names

    def name_for_module(self, module: nn.Module) -> str:
        if module not in self._module_names:
            raise KeyError(module)

        return self._module_names[module]

    def _new_name_for_module(self, module: nn.Module) -> str:
        # Count the number of the same type, to add suffix.
        module_type = type(module)
        self._type_count[module_type] += 1
        count = self._type_count[module_type]

        return f"{module_type.__name__}_{count}"


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
    The class that holds a `nn.Module`.
    """

    module: nn.Module
    args: tuple[typing.Any, ...]
    kwargs: dict[str, typing.Any]

    def __init__(self, module: nn.Module, *args, **kwargs):
        # Try applying `(*args, **kwargs)` to `module.forward` to see if it works.
        Sign.from_callable(module.forward).bind(*args, **kwargs)

        self.module = module
        self.args = args
        self.kwargs = kwargs

    def __repr__(self):
        return render_fcall(self.module, *self.args, **self.kwargs)

    @typing.override
    def compute(self, node_vals: AnyDict[BuilderNode, typing.Any], /) -> typing.Any:
        compute_node = lambda node: (
            node(node_vals) if isinstance(node, BuilderNode) else NotImplemented
        )

        args = decomp_replace(self.args, compute_node)
        kwargs = decomp_replace(self.kwargs, compute_node)

        return self.module(*args, **kwargs)

    def deps(self):
        yield from decomp_flatten(self.args, BuilderNode)
        yield from decomp_flatten(self.kwargs, BuilderNode)


class BuiltModule(nn.Module):
    "The module that traces from outputs to inputs with a pull strategy."

    def __init__(self, builder: CompoundBuilder, output: BuilderNode):
        super().__init__()

        self.sub_modules = nn.ModuleList([*builder.modules()])
        "The stored `nn.Module`s. This is kind of a hack to register the modules."

        self.builder: CompoundBuilder = builder
        "The builder itself."

        self.output: BuilderNode = output
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
    def _signature(self) -> Sign:
        return Sign.from_inputs([input.name for input in self.inputs])

    @property
    def inputs(self) -> list[InputBuilderNode]:
        return self.builder.inputs()

    @property
    def _input_names(self) -> dict[str, InputBuilderNode]:
        return {input.name: input for input in self.inputs}
