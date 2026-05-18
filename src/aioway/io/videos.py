# Copyright (c) AIoWay Authors - All Rights Reserved

import pathlib
import typing

import torch
from torchcodec import decoders as dec

from aioway._common import num_threads
from aioway.fake import torch_set_fake_mode_func

from .io import VideoLoader

__all__ = ["TorchCodecVideoLoader"]


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
