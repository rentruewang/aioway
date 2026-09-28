# Copyright (c) AIoWay Authors - All Rights Reserved

"A convenient visitor for `torch`."

import dataclasses as dcls
import types
import typing
from collections import abc as cabc

import loguru as L
import tensordict as td
import torch

__all__ = ["TorchVisitor"]


@dcls.dataclass(frozen=True)
class TorchVisitor[R: typing.Any = typing.Any]:
    """
    The visitor type for torch values.

    Using this to make sure we don't miss any values.
    """

    _: dcls.KW_ONLY

    tensor: cabc.Callable[[torch.Tensor], R] = NotImplemented
    "Tensor values."

    tdict: cabc.Callable[[td.TensorDict], R] = NotImplemented
    "TensorDict values."

    tcls: cabc.Callable[[typing.Any], R] = NotImplemented
    "Tensorclass values."

    mapping: cabc.Callable[[cabc.Mapping], R] = NotImplemented
    "Mapping values."

    sequence: cabc.Callable[[cabc.Sequence], R] = NotImplemented
    "Iterable values."

    default: cabc.Callable[[typing.Any], R] = NotImplemented
    "The default values."

    def __call__(self, item):
        if isinstance(item, torch.Tensor):
            return self._check_and_call(self.tensor)(item)

        if isinstance(item, td.TensorDict):
            return self._check_and_call(self.tdict)(item)

        if not isinstance(item, type) and td.is_tensor_collection(item):
            return self._check_and_call(self.tcls)(item)

        if isinstance(item, cabc.Mapping):
            return self._check_and_call(self.mapping)(item)

        if isinstance(item, cabc.Sequence):
            return self._check_and_call(self.sequence)(item)

        return self._check_and_call(self.default)(item)

    def _check_and_call[F: types.FunctionType](self, function: F) -> F:
        if function is not NotImplemented:
            return function

        name = function.__name__

        L.logger.error("Missing {} implementation for {}", name, self)
        raise TypeError(f"{self} did not implement {name}.")
