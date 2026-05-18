# Copyright (c) AIoWay Authors - All Rights Reserved

"The module that uses `av` to do audio and video processing."

import abc
import functools
import pathlib
import typing

import av
import numpy as np
from numpy import typing as npt

__all__ = ["AudioStream"]


class AudioStreamInfo(typing.NamedTuple):
    sample_rate: int
    num_channels: int
    duration: int

    @property
    def num_frames(self) -> int:
        return self.duration * self.sample_rate


class VideoStreamInfo(typing.NamedTuple):
    fps: float
    duration: int
    format: str
    width: int
    height: int

    @property
    def num_frames(self) -> int:
        return int(self.duration * self.fps)


class _AvStream[T](abc.ABC):
    def __init__(self, fname: str | pathlib.Path, /) -> None:
        self._fname = str(fname)

    @abc.abstractmethod
    def info(self) -> T:
        raise NotImplementedError

    @abc.abstractmethod
    def numpy(self) -> npt.NDArray[typing.Any]:
        raise NotImplementedError

    @functools.cached_property
    def container(self):
        return av.open(self._fname)

    @functools.cached_property
    def audio_stream(self):
        stream = self.container.streams.audio[0]
        stream.codec_context.thread_type = "AUTO"
        stream.codec_context.thread_count = 0
        return stream

    @functools.cached_property
    def video_stream(self):
        stream = self.container.streams.video[0]
        stream.codec_context.thread_type = "AUTO"
        stream.codec_context.thread_count = 0
        return stream


class AudioStream(_AvStream[AudioStreamInfo]):
    "The audio stream which is a light wrapper around `av.open` utilities."

    def info(self) -> AudioStreamInfo:
        stream = self.audio_stream
        duration = stream.duration
        assert duration, "Duration should exist, this is not a stream!"

        return AudioStreamInfo(
            sample_rate=stream.codec_context.sample_rate,
            num_channels=stream.codec_context.channels,
            duration=duration,
        )

    def numpy(self):
        chunks: list[np.ndarray] = []
        for frame in self.container.decode(self.audio_stream):
            arr = frame.to_ndarray()
            assert arr.ndim == 2
            chunks.append(arr)

        return np.concat(chunks, axis=1)


class VideoStream(_AvStream[VideoStreamInfo]):
    "The video decoder which is a light wrapper around `av.open` utilities."

    @typing.override
    def info(self) -> VideoStreamInfo:
        stream = self.video_stream
        fps = stream.average_rate
        assert fps

        duration = stream.duration
        assert duration

        format = stream.pix_fmt
        assert format

        return VideoStreamInfo(
            fps=float(fps),
            duration=duration,
            format=format,
            width=stream.width,
            height=stream.height,
        )

    def numpy(self):
        chunks: list[np.ndarray] = []
        for frame in self.container.decode(self.video_stream):
            # Would decode into an array of [h, w, 3].
            arr = frame.to_ndarray(format="rgb24")
            assert arr.ndim == 3
            chunks.append(arr)

        stacked = np.stack(chunks, axis=0)
        return np.einsum("nhwc->nchw", stacked)
