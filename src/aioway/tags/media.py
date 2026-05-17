# Copyright (c) AIoWay Authors - All Rights Reserved

"Extra information about the tensors."

import dataclasses as dcls

import torch

from aioway.fake import is_fake_tensor
from aioway.schemas import DType

from .tags import Tag

__all__ = ["IsImageTag", "SampleRateTag"]


@dcls.dataclass(frozen=True, slots=True)
class IsImageTag(Tag):
    """
    Tag the tensor as image. Should be 4 dimensional.
    """

    TAG = "__aioway_is_image__"

    def __post_init__(self) -> None:
        super().__post_init__()
        self._check_ndim()
        self._check_value()

    def _check_ndim(self):
        if self.tensor.ndim not in [3, 4]:
            raise ValueError(
                f"The tensor has ndim={self.tensor.ndim}!=3,4. Should be [B]CWH."
            )

    def _check_value(self):
        dtype = DType.parse(self.tensor.dtype)

        if dtype == torch.uint8:
            return

        # If fake mode then ok.
        if dtype.is_floating_point and is_fake_tensor(self.tensor):
            return

        def is_0_to_1():
            return (
                torch.all(0 <= self.tensor).item() and torch.all(self.tensor < 1).item()
            )

        if dtype.is_floating_point and is_0_to_1():
            return

        raise ValueError(
            f"The tensor with {dtype=} is not valid! "
            "Should be an uint8 tensor, or a floating point tensor with 0-1 values."
        )


@dcls.dataclass(frozen=True, slots=True)
class SampleRateTag(Tag):
    """
    Tag the tensor as audio, and having a sample rate.
    """

    TAG = "__aioway_audio_sample_rate__"

    sample_rate: int
    """
    The sample rate. Must be positive.
    """

    def __post_init__(self):
        if self.sample_rate <= 0:
            raise ValueError(f"{self.sample_rate} <= 0.")
