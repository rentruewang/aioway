# Copyright (c) AIoWay Authors - All Rights Reserved

import typing
from collections import abc as cabc

from torch.utils import data as dutils

from aioway.io import Stream

__all__ = ["AugStream"]


class AugStream[T](Stream[T]):
    """
    The `Stream` that "flat maps" over a dataset.
    """

    def __init__(
        self,
        dataset: dutils.Dataset[T],
        augment: cabc.Callable[[T], cabc.Generator[T]],
    ) -> None:
        self._dataset = dataset
        self._augment = augment

    @typing.no_type_check
    def __iter__(self) -> cabc.Generator[T]:
        for item in typing.cast(typing.Any, self._dataset):
            yield from self._augment(item)
