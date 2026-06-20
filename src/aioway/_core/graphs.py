# Copyright (c) AIoWay Authors - All Rights Reserved

"Metadata for torch operators / functions."

import typing
from collections import abc as cabc

import torch

__all__ = ["TensorInput"]


@typing.runtime_checkable
class TensorInput(typing.Protocol):
    """
    `TensorInput` marks a class whose value depend on input tensors for computation.
    """

    def inputs(self) -> cabc.Iterable[torch.Tensor]:
        "The tensor operands (inputs to the function)"

        raise NotImplementedError
