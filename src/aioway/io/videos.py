# Copyright (c) AIoWay Authors - All Rights Reserved

import os

import torch
from torchcodec import decoders as dec

from aioway.fake import torch_enable_fake_mode_func

__all__ = ["read_video_from_path"]


@torch_enable_fake_mode_func(False)
def read_video_from_path(fname: os.PathLike[str], threads: int = 1) -> torch.Tensor:
    "Read and decode videos from path."

    decoder = dec.VideoDecoder(str(fname), num_ffmpeg_threads=threads)
    samples = decoder[:]
    return samples
