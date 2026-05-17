# Copyright (c) AIoWay Authors - All Rights Reserved

import abc
import pathlib

import torch

__all__ = ["FileLoader", "ImageLoader", "AudioWaveLoader", "AudioFreqLoader"]


class FileLoader(abc.ABC):
    "The common file loader interface."

    @abc.abstractmethod
    def __call__(self, fname: str | pathlib.Path, /) -> torch.Tensor:
        raise NotImplementedError


class ImageLoader(FileLoader, abc.ABC):
    "The image loader API. Converts from a file name `fname` to a tensor."


class AudioWaveLoader(FileLoader, abc.ABC):
    "The audio loader in wave form."


class AudioFreqLoader(FileLoader, abc.ABC):
    "The audio loader in frequency form."
