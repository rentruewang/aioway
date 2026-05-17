# Copyright (c) AIoWay Authors - All Rights Reserved

import pathlib
import typing

import av
import numpy as np
import torch
from torchcodec import decoders as dec

from aioway.fake import enabled_fake_mode, torch_enable_fake_mode_func
from aioway.schemas import Attr
from aioway.tags import SampleRateTag

from .io import AudioLoader

__all__ = ["AvAudioLoader", "TorchCodecAudioLoader"]


def encode_with_stft(audio: torch.Tensor, /, n_fft: int) -> torch.Tensor:
    """
    Encode the audio with `torch.stft`. Naturally works with fake mode.
    """

    result = torch.stft(audio, n_fft, return_complex=True)
    SampleRateTag.extract(audio)
    return result.real


class AvAudioLoader(AudioLoader):
    "Load the audio with `av` library."

    @typing.override
    def load_wave(self, fname: str | pathlib.Path, /) -> torch.Tensor:
        container = av.open(fname)

        # Get the metadata for fake mode to work.
        stream = container.streams.audio[0]
        sample_rate = stream.codec_context.sample_rate
        channels = stream.codec_context.channels
        assert stream.duration, "Duration should exist, this is not a stream!"
        frames = stream.duration * stream.sample_rate

        # Create a fake tensor of float32 in fake mode.
        if enabled_fake_mode():
            tensor = Attr.parse(
                shape=[channels, frames], dtype=torch.float32
            ).to_fake_tensor()

        # Decode frame by frame.
        else:
            chunks: list[np.ndarray] = []
            for frame in container.decode(stream):
                arr = frame.to_ndarray()
                assert arr.ndim == 2
                assert len(arr) == channels
                chunks.append(arr)

            array = np.concat(chunks, axis=1)
            tensor = torch.tensor(array)

        _ = SampleRateTag(tensor, sample_rate)
        return tensor


class TorchCodecAudioLoader(AudioLoader):

    @typing.override
    def load_wave(self, fname: str | pathlib.Path, /) -> torch.Tensor:
        return self._read_audio_from_path(fname)

    @torch_enable_fake_mode_func(False)
    def _read_audio_from_path(self, fname: str | pathlib.Path) -> torch.Tensor:
        """
        Read and decode audio from path.

        Args:
            fname: The file location.
            sample_rate:
                The sample rate in Hz.
                If given, this should be equal to the sample rate in return value.
        """

        decoder = dec.AudioDecoder(str(fname), sample_rate=self.sample_rate)
        samples = decoder.get_all_samples()

        if self.sample_rate and self.sample_rate != samples.sample_rate:
            raise AssertionError(
                f"The configured sample rates {self.sample_rate} "
                f"and output {samples.sample_rate} should be equal."
            )

        _ = SampleRateTag(samples.data, samples.sample_rate)
        return samples.data
