# Copyright (c) AIoWay Authors - All Rights Reserved

import abc
import dataclasses as dcls
import typing
from collections import abc as cabc

from aioway._utils import decomp_flatten
from aioway.builders import TensorBuilder
from aioway.hop import Hop, ListHop, TensorHop
from aioway.nn import Linear
from aioway.spaces import ShapeSpace, Space

__all__ = ["Emitter", "emitter_dcls", "JustLinearEmitter"]


@typing.dataclass_transform(frozen_default=True)
def emitter_dcls(cls):
    return dcls.dataclass(frozen=True)(cls)


@emitter_dcls
class Emitter(abc.ABC):
    """
    `Emitter` emits an additional `Hop` that can be added on top of the current `Hop` network,
    providing additional transformation on top of what we currently have.

    An `Emitter` should raise an error in `__post_init__` if the inputs are wrong.
    """

    @abc.abstractmethod
    def __call__(self) -> Hop:
        """
        Compiles from a list of input and output tags.
        """

        raise NotImplementedError

    def inputs(self) -> cabc.Iterator[Hop]:
        """
        In the current version of the design, an `Emitter` adds something to a `Hop`.
        Therefore we could store the input `Hop` onto itself.
        """

        yield from decomp_flatten(self, Hop)

    def outputs(self) -> cabc.Iterator[Space]:
        """
        In the current version of the design,
        an `Emitter` would store the output spaces onto itself.
        """

        yield from decomp_flatten(self, Space)


@emitter_dcls
class JustLinearEmitter(Emitter):
    """
    A builder that outputs 1 linear layer, supporting 1 input and 1 output.
    """

    input: TensorHop
    "The input `Hop`."

    output: ShapeSpace
    "The output contract."

    def __post_init__(self) -> None:
        assert isinstance(self.input, TensorHop)
        assert isinstance(self.output, ShapeSpace)

    def __call__(self) -> ListHop:
        input_node = TensorBuilder(self.input)
        linear_node = input_node.apply_layer(
            Linear(self.input.attr.shape[-1], self.output[-1]),
        )
        return ListHop([linear_node.hop])
