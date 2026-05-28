# Copyright (c) AIoWay Authors - All Rights Reserved

import typing

from aioway._torch import torch_set_fake_mode_func
from aioway.hop import HopDag, Linear, TensorHop
from aioway.tags import Tag

__all__ = ["Builder", "just_linear_builder"]


class Builder(typing.Protocol):
    def __call__(self, inputs: list[Tag], outputs: list[Tag]) -> HopDag:
        """
        Compiles from a list of input and output tags.
        Returns `NotImplemented` if the current builder does not support the inputs outputs.
        """

        ...


@torch_set_fake_mode_func(True)
def just_linear_builder(inputs: list[Tag], outputs: list[Tag]) -> HopDag:
    try:
        input_schema = _check_linear_io(inputs)
        output_schema = _check_linear_io(outputs)
    except NotImplementedError:
        return NotImplemented

    if input_schema.shape[:-1] != output_schema.shape[:-1]:
        return NotImplemented

    input_node = TensorHop(input_schema.to_fake_tensor())

    linear_layer = Linear(input_schema.shape[-1], output_schema.shape[-1])
    linear_node = linear_layer.apply(input_node)

    return HopDag.from_list_of_nodes([input_node, linear_node])


def _check_linear_io(spaces: list[Tag]):
    if len(spaces) != 1:
        raise NotImplementedError

    [space] = spaces

    if not isinstance(space, SchemaSpace):
        raise NotImplementedError

    if space.schema.attr.dtype.family != "float":
        raise NotImplementedError

    return space.schema
