# Copyright (c) AIoWay Authors - All Rights Reserved

import abc
import dataclasses as dcls
import pathlib
import typing

import torch

from aioway.tags import IsImageTag, IsVideoTag, SampleRateTag

__all__ = ["FileLoader", "ImageLoader", "AudioLoader", "VideooLoader"]


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

        IsImageTag().attach(data)

        return data

    @abc.abstractmethod
    def load_img(self, fname: str | pathlib.Path, /) -> torch.Tensor:
        raise NotImplementedError


class AudioData(typing.NamedTuple):
    data: torch.Tensor
    sample_rate: int


@dcls.dataclass
class AudioLoader(FileLoader, abc.ABC):
    """
    The audio loader API. Load the data into wave.
    Result is tensor [num_channels, num_frames].
    """

    @typing.override
    def __call__(self, fname: str | pathlib.Path, /) -> torch.Tensor:
        audio = self.load_wave(fname)

        SampleRateTag(audio.sample_rate).attach(audio.data)

        assert audio.data.ndim == 2, audio.data.shape
        return audio.data

    @abc.abstractmethod
    def load_wave(self, fname: str | pathlib.Path, /) -> AudioData:
        raise NotImplementedError


@dcls.dataclass
class VideooLoader(FileLoader, abc.ABC):
    """
    The video loader API. Load the data into a 4D tensor.
    """

    @typing.override
    def __call__(self, fname: str | pathlib.Path, /) -> torch.Tensor:
        video = self.load_video(fname)

        IsVideoTag().attach(video)

        return video

    @abc.abstractmethod
    def load_video(self, fname: str | pathlib.Path, /) -> torch.Tensor:
        raise NotImplementedError
