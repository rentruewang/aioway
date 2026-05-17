# Copyright (c) AIoWay Authors - All Rights Reserved

"Extra information about the tensors."

import typing

import torch

from aioway._common import dcls_frozen_slots
from aioway.fake import is_fake_tensor
from aioway.schemas import DType

from .tags import Tag

__all__ = ["IsImageTag", "SampleRateTag"]


@dcls_frozen_slots
class IsImageTag(Tag):
    """
    Tag the tensor as image. Should be 4 dimensional.
    """

    TAG = "__aioway_is_image__"

    @typing.override
    def _validate(self, tensor: torch.Tensor) -> None:
        self._check_ndim(tensor)
        self._check_value(tensor)

    def _check_ndim(self, tensor: torch.Tensor):
        if tensor.ndim not in [3, 4]:
            raise ValueError(
                f"The tensor has ndim={tensor.ndim}!=3,4. Should be [B]CWH."
            )

    def _check_value(self, tensor: torch.Tensor):
        dtype = DType.parse(tensor.dtype)

        if dtype == torch.uint8:
            return

        # If fake mode then ok.
        if dtype.is_floating_point and is_fake_tensor(tensor):
            return

        def is_0_to_1():
            return torch.all(0 <= tensor).item() and torch.all(tensor < 1).item()

        if dtype.is_floating_point and is_0_to_1():
            return

        raise ValueError(
            f"The tensor with {dtype=} is not valid! "
            "Should be an uint8 tensor, or a floating point tensor with 0-1 values."
        )


@dcls_frozen_slots
class SampleRateTag(Tag):
    """
    Tag the tensor as audio, and having a sample rate.
    """

    TAG = "__aioway_audio_sample_rate__"

    sample_rate: int
    """
    The sample rate. Must be positive.
    """

    @typing.override
    def _validate(self, tensor: torch.Tensor) -> None:
        if self.sample_rate <= 0:
            raise ValueError(f"{self.sample_rate} <= 0.")
