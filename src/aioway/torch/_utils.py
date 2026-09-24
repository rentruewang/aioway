# Copyright (c) AIoWay Authors - All Rights Reserved

import dataclasses as dcls

import tensordict as td

__all__ = ["tcol_to_tdict"]


def tcol_to_tdict(item) -> td.TensorDict:
    "Convert from tensor collection to `TensorDict`."

    if not td.is_tensor_collection(item):
        raise ValueError("Not tensor collection.")

    if isinstance(item, td.TensorDict):
        return item

    assert dcls.is_dataclass(item)
    attrs = dcls.asdict(item)
    result = td.from_dict(attrs)
    assert isinstance(result, td.TensorDict)
    return result
