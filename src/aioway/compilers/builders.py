# Copyright (c) AIoWay Authors - All Rights Reserved

import abc
import dataclasses as dcls
import typing
from collections import abc as cabc

from aioway.hop import HopGraph, Linear, TensorHop
from aioway.tags import AttrTag, TagDict

__all__ = ["Builder", "builder_dcls", "JustLinearBuilder"]


@typing.dataclass_transform(frozen_default=True)
def builder_dcls(cls):
    return dcls.dataclass(frozen=True)(cls)


@builder_dcls
class Builder(abc.ABC):
    @abc.abstractmethod
    def __call__(self) -> HopGraph:
        """
        Compiles from a list of input and output tags.
        Returns `NotImplemented` if the current builder does not support the inputs outputs.
        """

        raise NotImplementedError

    @abc.abstractmethod
    def inputs(self) -> cabc.Iterator[TagDict]:
        raise NotImplementedError

    @abc.abstractmethod
    def outputs(self) -> cabc.Iterator[TagDict]:
        raise NotImplementedError


@builder_dcls
class JustLinearBuilder(Builder):
    """
    A builder that outputs 1 linear layer, supporting 1 input and 1 output.
    """

    in_attr: AttrTag
    "The input shape."

    out_attr: AttrTag
    "The output shape."

    def __call__(self) -> HopGraph:
        try:
            in_attr = self.in_attr.to_attr()
            out_attr = self.out_attr.to_attr()
        except TypeError:
            return NotImplemented

        input_node = TensorHop(in_attr.to_fake_tensor())

        linear_layer = Linear(in_attr.shape[-1], out_attr.shape[-1])
        linear_node = linear_layer.apply(input_node)
        return HopGraph(linear_node)

    def inputs(self) -> cabc.Iterator[TagDict]:
        yield TagDict.from_tags(self.in_attr)

    def outputs(self) -> cabc.Iterator[TagDict]:
        yield TagDict.from_tags(self.out_attr)
