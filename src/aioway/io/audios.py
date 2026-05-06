# Copyright (c) AIoWay Authors - All Rights Reserved

import os
import typing

import torch
from torchcodec import decoders as dec

from aioway.fn import torch_enable_fake_mode_func

__all__ = ["AudioData", "read_audio_from_path"]


class AudioData(typing.Protocol):
    data: torch.Tensor
    "The sample data, [num channels, num samples]."

    sample_rate: int
    "The sample rate in Hz."


@torch_enable_fake_mode_func(False)
def read_audio_from_path(
    fname: os.PathLike[str], sample_rate: int | None = None
) -> AudioData:
    """
    Read and decode audio from path.

    Args:
        fname: The file location.
        sample_rate:
            The sample rate in Hz.
            If given, this should be equal to the sample rate in return value.
    """

    decoder = dec.AudioDecoder(str(fname), sample_rate=sample_rate)
    samples = decoder.get_all_samples()

    if sample_rate and sample_rate != samples.sample_rate:
        raise AssertionError(
            f"The configured sample rates {sample_rate} "
            f"and output {samples.sample_rate} should be equal."
        )

    return samples
