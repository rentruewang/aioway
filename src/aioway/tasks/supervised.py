# Copyright (c) AIoWay Authors - All Rights Reserved

import typing
from .tasks import Task, NnInput, BatchIter


class SupervisedTask[T: NnInput](Task[T]):
    @typing.override
    def fake(self) -> T:
        raise NotImplementedError

    @typing.override
    def iterator(self) -> BatchIter[T]:
        raise NotImplementedError

    def step(self, batch: T) -> None:
        raise NotImplementedError
