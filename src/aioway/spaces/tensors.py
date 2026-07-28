# Copyright (c) AIoWay Authors - All Rights Reserved

"Some common tensor operations."

from torch import nn

from aioway.attrs import Shape

from .attrs import ShapeSpace
from .casts import CastedSpaceModule, register_cast

__all__ = ["flatten_tensor"]


@register_cast(ShapeSpace, ShapeSpace)
def flatten_tensor(input_space: ShapeSpace) -> CastedSpaceModule:
    flattened = input_space.shape.numel()
    output_space = ShapeSpace(Shape.parse(flattened))
    return CastedSpaceModule(output_space, module=nn.Flatten())
