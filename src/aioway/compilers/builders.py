# Copyright (c) AIoWay Authors - All Rights Reserved

import typing

from aioway.hop import HopInit, Linear, TensorHopInit
from aioway.spaces import SchemaSpace, Space

__all__ = ["Builder", "just_linear_builder"]


class Builder(typing.Protocol):
    def __call__(self, inputs: list[Space], outputs: list[Space]) -> list[HopInit]:
        """
        Compiles from a list of input and output spaces.
        Returns `NotImplemented` if the current builder does not support the inputs outputs.
        """

        ...


def just_linear_builder(inputs: list[Space], outputs: list[Space]) -> list[HopInit]:
    try:
        input_schema = _check_linear_io(inputs)
        output_schema = _check_linear_io(outputs)
    except NotImplementedError:
        return NotImplemented

    if input_schema.shape[:-1] != output_schema.shape[:-1]:
        return NotImplemented

    input_node = TensorHopInit(tensor=input_schema.to_fake_tensor())
    linear_init = Linear(input_schema.shape[-1], output_schema.shape[-1])
    linear_node = linear_init.apply_hop(input_node)

    return [input_node, linear_node]


def _check_linear_io(spaces: list[Space]):
    if len(spaces) != 1:
        raise NotImplementedError

    [space] = spaces

    if not isinstance(space, SchemaSpace):
        raise NotImplementedError

    if space.schema.attr.dtype.family != "float":
        raise NotImplementedError

    return space.schema
