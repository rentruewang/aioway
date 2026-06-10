# Copyright (c) AIoWay Authors - All Rights Reserved

"The dataset related `Hop`s."

import typing

from aioway.relalg import TensorHop

from .hop import Hop, hop_dcls

__all__ = ["TensorStreamHop"]


@hop_dcls
class TensorStreamHop(Hop):
    stream: TensorHop

    def iterate(self):
        yield from self.stream

    @property
    @typing.override
    def requires_grad(self) -> bool:
        return False
