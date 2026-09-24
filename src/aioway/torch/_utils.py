# Copyright (c) AIoWay Authors - All Rights Reserved

import dataclasses as dcls
import typing

import tensordict as td

__all__ = ["tcol_to_tdict", "dcls_asdict"]


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


def dcls_asdict(obj: object) -> dict[str, typing.Any]:
    "Official `asdict` fail with some custom `__getstate__`s."

    assert dcls.is_dataclass(obj), "Only handles dataclass objects."
    fields = dcls.fields(obj)
    return {field.name: getattr(obj, field.name) for field in fields}
