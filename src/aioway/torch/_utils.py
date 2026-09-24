# Copyright (c) AIoWay Authors - All Rights Reserved

import dataclasses as dcls
from collections import abc as cabc

import tensordict as td
import torch

from aioway._utils import dcls_asdict, tree_map_memo

__all__ = ["tcol_to_tdict"]


def tcol_to_tdict(item) -> td.TensorDict:
    "Convert from tensor collection to `TensorDict`."

    if not td.is_tensor_collection(item):
        raise ValueError("Not tensor collection.")

    if isinstance(item, td.TensorDict):
        return item

    assert dcls.is_dataclass(item)
    attrs = dcls_asdict(item)
    result = td.from_dict(attrs)
    assert isinstance(result, td.TensorDict)
    return result


def replace_tensors(
    obj: object, replace: cabc.Callable[[torch.Tensor], object]
) -> object:
    """
    Replace tensors whenever encountered with the given function.

    This function has the `__torch_function__` disabled in the scope of the rendering,
    because it can mess with attribute access, which oftentimes means that
    this function fails also during debugging if `__torch_function__` is not disabled.
    Caused by `.device` / `.shape` / `.dtype` calls, which is used in `replace_tensors`.
    """

    from .overrides import mode_off

    def maybe_replace(item):
        if not isinstance(item, torch.Tensor):
            return NotImplemented

        return replace(item)

    with mode_off():
        return tree_map_memo(obj, maybe_replace)
