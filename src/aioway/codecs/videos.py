# Copyright (c) AIoWay Authors - All Rights Reserved

import abc
import dataclasses as dcls
import pathlib
import typing

import torch
from torchcodec import decoders as dec

from aioway._utils import current_fake_mode, num_threads, torch_set_fake_mode_func
from aioway.attrs import Attr

from ._av import VideoStream
from ._bases import TorchCompatible

__all__ = ["VideoLoader", "AvVideoLoader", "TorchCodecVideoLoader", "VideoData"]


@dcls.dataclass(frozen=True)
class VideoData(TorchCompatible):
    data: torch.Tensor
    "The tensor decoded from the video."

    @typing.override
    def to_tensor(self) -> torch.Tensor:
        return self.data


@dcls.dataclass
class VideoLoader(abc.ABC):
    """
    The video loader API. Load the data into a 4D tensor.
    """

    def __call__(self, fname: str | pathlib.Path, /) -> VideoData:
        video = self.load_video(fname)
        return VideoData(video)

    @abc.abstractmethod
    def load_video(self, fname: str | pathlib.Path, /) -> torch.Tensor:
        raise NotImplementedError


class AvVideoLoader(VideoLoader):
    "Load the video with `av` library."

    @typing.override
    def load_video(self, fname: str | pathlib.Path, /) -> torch.Tensor:
        stream = VideoStream(fname)

        # Get the metadata for fake mode to work.
        info = stream.info()

        # Create a fake tensor of float32 in fake mode.
        if current_fake_mode():
            tensor = Attr.build(
                shape=[info.num_frames, 3, info.width, info.height], dtype=torch.float32
            ).to_fake_tensor()

        else:
            array = stream.numpy()
            assert array.ndim == 4
            assert array.shape[1] == 3
            tensor = torch.from_numpy(array)

        return tensor


class TorchCodecVideoLoader(VideoLoader):
    """
    Load the video from a path with `torchcodec` library.

    Note that similar to `TorchCodecAudioLoader`, even during fake mode,
    the video is still loaded in memory as `torch.Tensor`,
    because there aren't any good way to overwrite torch codecs to enable fake mode.
    This is due to limitations in `torchcodecs` as they use custom `torch.ops`.
    """

    threads: int = num_threads(8)
    """
    Number of `ffmpeg` threads to use.
    """

    @typing.override
    @torch_set_fake_mode_func(False)
    def load_video(self, fname: str | pathlib.Path, /) -> torch.Tensor:
        "Read and decode videos from path."

        decoder = dec.VideoDecoder(str(fname), num_ffmpeg_threads=self.threads)
        samples = decoder[:]
        return samples
