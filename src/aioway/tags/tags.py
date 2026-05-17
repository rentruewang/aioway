# Copyright (c) AIoWay Authors - All Rights Reserved

"Extra information about the tensors."

import abc
import dataclasses as dcls
import typing

import torch

from aioway.fake import is_fake_tensor

if typing.TYPE_CHECKING:
    from .audios import SampleRateTag
    from .dims import DimTag

__all__ = ["Tag", "DimTagged"]


@dcls.dataclass(frozen=True, slots=True)
class Tag(abc.ABC):
    """
    The base class for tags.
    """

    tensor: torch.Tensor
    "The tensor that is being piggybacked."

    @property
    def ndim(self) -> int:
        return self.tensor.ndim

    @property
    def shape(self) -> torch.Size:
        return self.tensor.shape

    @property
    def is_fake(self) -> bool:
        return is_fake_tensor(self.tensor)


@typing.runtime_checkable
class DimTagged(typing.Protocol):
    __aioway_dim_tag__: DimTag
    "The dimension tag on the tensor."


@typing.runtime_checkable
class SampleRateTagged(typing.Protocol):
    __aioway_sample_rate__: SampleRateTag
    "The sample rate on the audio tensor."
