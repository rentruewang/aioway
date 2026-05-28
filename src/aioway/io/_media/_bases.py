# Copyright (c) AIoWay Authors - All Rights Reserved

"The common utilities."

import typing

import torch

__all__ = ["TorchCompatible"]


class TorchCompatible(typing.Protocol):
    def to_tensor(self) -> torch.Tensor: ...
