# Copyright (c) AIoWay Authors - All Rights Reserved

"Some common tensor operations."

import torch

from aioway._ufuncs import AdHocUFunc
from aioway.attrs import Shape

from .casts import CastedSpaceUFunc, register_cast
from .exact import ShapeSpace

__all__ = ["flatten_tensor"]


@register_cast(ShapeSpace, ShapeSpace)
def flatten_tensor(input_space: ShapeSpace) -> CastedSpaceUFunc:
    flattened = input_space.shape.numel()
    output_space = ShapeSpace(Shape.parse(flattened))
    return CastedSpaceUFunc(output_space, ufunc=AdHocUFunc(torch.flatten))
