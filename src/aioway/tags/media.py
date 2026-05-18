# Copyright (c) AIoWay Authors - All Rights Reserved

"Extra information about the tensors."

import typing
from collections import abc as cabc

import torch

from aioway._common._types import dcls_frozen_slots
from aioway.fake import is_fake_tensor
from aioway.schemas import DType

from .tags import Tag

__all__ = ["IsTokenizedTag", "IsVideoTag", "IsImageTag", "SampleRateTag"]


@dcls_frozen_slots
class IsTokenizedTag(Tag):
    """
    Tag the tensor as embeddings (tokenized).
    """

    TAG = "__aioway_is_tokenized__"

    tokenizer: str
    """
    The name of the tokenizer used to construct the tensor.
    """

    @typing.override
    def _validate(self, tensor: torch.Tensor) -> None:
        if (family := DType.parse(tensor.dtype).family) != "int":
            raise ValueError(
                f"Tokenized result has dtype family: '{family}', not 'int'."
            )


@dcls_frozen_slots
class IsVideoTag(Tag):
    """
    Tag the tensor as video. Must be 5, 5 dimensional (with or without batch).
    """

    TAG = "__aioway_is_video__"

    @typing.override
    def _validate(self, tensor: torch.Tensor) -> None:
        _check_image_or_video(tensor, _VIDEO_NDIM_INFO)


@dcls_frozen_slots
class IsImageTag(Tag):
    """
    Tag the tensor as image. Should be 3, 4 dimensional (with or without batch).
    """

    TAG = "__aioway_is_image__"

    @typing.override
    def _validate(self, tensor: torch.Tensor) -> None:
        _check_image_or_video(tensor, _IMAGE_NDIM_INFO)


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


class NdimInfo(typing.NamedTuple):
    "The video / image ndim related info."

    valid_ndims: cabc.Sequence[int]
    "The valid ndims."

    channels: str
    "The names for the channels."


_IMAGE_NDIM_INFO = NdimInfo(valid_ndims=[3, 4], channels="[N]CHW")
_VIDEO_NDIM_INFO = NdimInfo(valid_ndims=[4, 5], channels="[N]CTHW")


def _check_image_or_video(tensor: torch.Tensor, info: NdimInfo):
    if tensor.ndim not in info.valid_ndims:
        raise ValueError(
            f"The tensor has ndim={tensor.ndim}. Should be {info.channels}."
        )

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
