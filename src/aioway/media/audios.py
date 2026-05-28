# Copyright (c) AIoWay Authors - All Rights Reserved

import abc
import dataclasses as dcls
import pathlib
import typing

import torch
from torch.utils import data
from torchcodec import decoders as dec

from aioway._torch import current_fake_mode, torch_set_fake_mode_func
from aioway.schemas import Attr, IsStftTag, SampleRateTag

from ._av import AudioStream
from ._bases import TorchCompatible
from .collates import chunk_collate

__all__ = [
    "AudioLoader",
    "AudioDataFolder",
    "AvAudioLoader",
    "TorchCodecAudioLoader",
    "AudioData",
    "encode_with_stft",
]


@dcls.dataclass(frozen=True)
class AudioData(TorchCompatible):
    data: torch.Tensor
    "The loaded audio."

    sample_rate: int
    "The sample rate."

    def __post_init__(self):
        assert self.data.ndim == 2, self.data.shape

    @typing.override
    def to_tensor(self):
        SampleRateTag(self.sample_rate).attach(self.data)
        return self.data


@dcls.dataclass
class AudioLoader(abc.ABC):
    """
    The audio loader API. Load the data into wave.
    Result is tensor [num_channels, num_frames].
    """

    sample_rate: int | None = None
    """
    The sample rate to choose. If `None` use the audio's default.
    """

    @abc.abstractmethod
    def __call__(self, fname: str | pathlib.Path, /) -> AudioData:
        raise NotImplementedError


def encode_with_stft(audio: torch.Tensor, /, n_fft: int) -> torch.Tensor:
    """
    Encode the audio with `torch.stft`. Naturally works with fake mode.
    """

    result = torch.stft(audio, n_fft, return_complex=True)
    real_result = result.real

    # Perhaps we should handle tags passing in `Fate`.
    if tag := SampleRateTag.extract(audio):
        tag.attach(real_result)

    return real_result


class AudioDataFolder(data.Dataset[torch.Tensor]):
    def __init__(self, loader: AudioLoader, *files: str) -> None:
        super().__init__()
        self._files = files
        self._loader = loader

        if not self._loader.sample_rate:
            raise ValueError("Sample rate must be given.")

    def __len__(self) -> int:
        return len(self._files)

    def __getitem__(self, idx: int) -> torch.Tensor:
        return self._loader(self._files[idx]).to_tensor()

    def collate(self, max_len: int):
        """
        The collate function that chunks the input and re-batch them.

        Args:
            max_len: The maximum length in the frame dimension to chunk.
        """

        collate = chunk_collate(max_len)

        def collate_and_tag(audios: list[torch.Tensor]):
            # Since audio has shape [num_channels, num_frames],
            # need to permute it for it to work.
            audios = [audio for audio in audios]
            result = collate(audios)
            SampleRateTag(sample_rate=self.sample_rate).attach(result)
            return result

        return collate_and_tag

    def collate_stft(self, max_len: int, n_fft: int):
        """
        Collate into stft (after chunking with `chunk_stft` for simplcity).
        """

        collate = self.collate(max_len)

        def do_stft(audios: list[torch.Tensor]) -> torch.Tensor:
            tensor = collate(audios)
            stft = encode_with_stft(tensor, n_fft=n_fft)
            SampleRateTag(sample_rate=self.sample_rate).attach(stft)
            IsStftTag().attach(stft)
            return stft

        return do_stft

    @property
    def sample_rate(self) -> int:
        assert self._loader.sample_rate
        return self._loader.sample_rate


class AvAudioLoader(AudioLoader):
    "Load the audio with `av` library."

    @typing.override
    def __call__(self, fname: str | pathlib.Path, /) -> AudioData:
        stream = AudioStream(fname, sample_rate=self.sample_rate)

        # Get the metadata for fake mode to work.
        info = stream.info()

        # Create a fake tensor of float32 in fake mode.
        if current_fake_mode():
            tensor = Attr.build(
                shape=[info.num_channels, info.num_frames], dtype=torch.float32
            ).to_fake_tensor()

        # Decode frame by frame.
        else:
            array = stream.numpy()
            assert len(array) == info.num_channels
            tensor = torch.from_numpy(array)

        return AudioData(tensor, stream.sample_rate)


class TorchCodecAudioLoader(AudioLoader):
    """
    The `AudioLoader` backed by the `torchcodec` library.

    Note that similar to `TorchCodecVideoLoader`, even during fake mode,
    the video is still loaded in memory as `torch.Tensor`,
    because there aren't any good way to overwrite torch codecs to enable fake mode.
    This is due to limitations in `torchcodecs` as they use custom `torch.ops`.
    """

    @typing.override
    def __call__(self, fname: str | pathlib.Path, /) -> AudioData:
        return self._read_audio_from_path(fname)

    @torch_set_fake_mode_func(False)
    def _read_audio_from_path(self, fname: str | pathlib.Path) -> AudioData:
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

        return AudioData(samples.data, samples.sample_rate)
