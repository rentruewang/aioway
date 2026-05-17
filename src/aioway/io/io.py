# Copyright (c) AIoWay Authors - All Rights Reserved

import abc
import dataclasses as dcls
import pathlib
import typing

import torch

from aioway.tags import IsImageTag
from aioway.tags.media import SampleRateTag


@dcls.dataclass
class FileLoader(abc.ABC):
    "The base loader API."

    @abc.abstractmethod
    def __call__(self, fname: str | pathlib.Path, /) -> torch.Tensor:
        raise NotImplementedError


@dcls.dataclass
class ImageLoader(FileLoader, abc.ABC):
    "The image loader API. Converts from a file name `fname` to a tensor."

    @typing.override
    def __call__(self, fname: str | pathlib.Path, /) -> torch.Tensor:
        data = self.load_img(fname)

        # Attach the tag onto the tensor.
        _ = IsImageTag(data)

        return data

    @abc.abstractmethod
    def load_img(self, fname: str | pathlib.Path, /) -> torch.Tensor:
        raise NotImplementedError


@dcls.dataclass
class AudioLoader(FileLoader, abc.ABC):
    """
    The audio loader API. Load the data into wave.
    Result is tensor [num_channels, num_frames].
    """

    sample_rate: int | None = None
    "The specified sample rate. If `None` the default would be loaded."

    @typing.override
    def __call__(self, fname: str | pathlib.Path, /) -> torch.Tensor:
        audio = self.load_wave(fname)

        # Ensure it's actually tagged.
        if SampleRateTag.extract(audio) is None:
            raise AssertionError(f"Forgot to tag the audio tensor with sample rate.")

        return audio

    @abc.abstractmethod
    def load_wave(self, fname: str | pathlib.Path, /) -> torch.Tensor:
        raise NotImplementedError
