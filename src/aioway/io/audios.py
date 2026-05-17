# Copyright (c) AIoWay Authors - All Rights Reserved

import pathlib
import typing

import torch
from torchcodec import decoders as dec

from aioway.fake import torch_enable_fake_mode_func
from aioway.tags import SampleRateTag

from .io import AudioLoader

__all__ = ["TorchCodecAudioLoader"]


def encode_with_stft(audio: torch.Tensor, /, n_fft: int) -> torch.Tensor:
    """
    Encode the audio with `torch.stft`. Naturally works with fake mode.
    """

    result = torch.stft(audio, n_fft, return_complex=True)
    SampleRateTag.extract(audio)
    return result.real


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
