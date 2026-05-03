# Copyright (c) AIoWay Authors - All Rights Reserved

import os
import typing

import torch
from torchcodec import decoders as dec

__all__ = ["AudioData", "read_audio_from_path"]


class AudioData(typing.Protocol):
    data: torch.Tensor
    "The sample data, [num channels, num samples]."

    sample_rate: float
    "The sample rate in Hz."


def read_audio_from_path(fname: os.PathLike[str]):
    decoder = dec.AudioDecoder(str(fname))
    samples = decoder.get_all_samples()
    return samples
