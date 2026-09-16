# Copyright (c) AIoWay Authors - All Rights Reserved


import typing

from aioway.tensors import as_tspec, sample_from_tspec

from .tasks import BatchIter, NnInput, Task


class SupervisedTask[T: NnInput](Task[T]):
    batch_iter: BatchIter

    @typing.override
    def fake(self) -> T:
        return sample_from_tspec(as_tspec(self.batch_iter.__tspec__()))

    @typing.override
    def iterator(self) -> BatchIter[T]:
        return self.batch_iter

    def step(self, batch: T) -> None:
        input, target = batch
        raise NotImplementedError
