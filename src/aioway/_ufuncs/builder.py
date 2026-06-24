# Copyright (c) AIoWay Authors - All Rights Reserved

import abc
import collections
import dataclasses as dcls
import inspect
import typing
from collections import abc as cabc

from aioway._utils import AnySet, decomp_flatten, decomp_replace

from .ufuncs import UFunc

__all__ = ["BuilderNode", "InputBuilderNode", "ThunkBuilderNode", "BuiltUFunc"]


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

    def build(self) -> UFunc:
        """
        Build a `UFunc` based on the `output` node and the `inputs` nodes stored in `self`.
        """

        all_nodes = AnySet[BuilderNode]()

        def visit_node(node: BuilderNode):
            if node in all_nodes:
                return

            all_nodes.add(node)

            for dep in node.deps():
                visit_node(dep)

        visit_node(self)

        # Get the traced input, and convert them to **kwargs
        inputs = [node for node in all_nodes if isinstance(node, InputBuilderNode)]

        # Check for duplicates.
        name_count: dict[str, int] = collections.defaultdict(int)
        for input in inputs:
            name_count[input.name] += 1
        if dups := {name: count for name, count in name_count.items() if count > 1}:
            raise KeyError(f"Duplicate keys found: {list(dups.keys())}.")

        return BuiltUFunc(inputs, self)


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
    inputs: list[InputBuilderNode]
    output: BuilderNode

    def forward(self, *args, **kwargs):
        signature = self.__signature__
        bound = signature.bind(*args, **kwargs)
        bound.apply_defaults()
        all_kwargs = bound.arguments
        assert isinstance(all_kwargs, dict), all_kwargs
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
