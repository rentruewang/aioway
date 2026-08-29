# Copyright (c) AIoWay Authors - All Rights Reserved

import tensordict as td
from torchrl.data import tensor_specs as tspecs

__all__ = ["tspec_from_tclass"]


def tspec_from_tclass(composed_type: type[td.TensorClass]) -> tspecs.Composite:
    raise NotImplementedError
