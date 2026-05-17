# Copyright (c) AIoWay Authors - All Rights Reserved

import abc
import dataclasses as dcls
import pathlib
import typing

import torch

from aioway.tags import IsImageTag


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
        data = self.load(fname)

        # Attach the tag onto the tensor.
        _ = IsImageTag(data)

        return data

    @abc.abstractmethod
    def load(self, fname: str | pathlib.Path, /) -> torch.Tensor:
        raise NotImplementedError
