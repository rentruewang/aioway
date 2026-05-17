# Copyright (c) AIoWay Authors - All Rights Reserved

"Extra information about the tensors."

import dataclasses as dcls

from .tags import Tag

__all__ = ["SampleRateTag"]


@dcls.dataclass(frozen=True)
class SampleRateTag(Tag):
    """
    Tag the tensor as audio, and having a sample rate.
    """

    sample_rate: int
    """
    The sample rate. Must be positive.
    """

    def __post_init__(self):
        super().__post_init__()

        if self.sample_rate <= 0:
            raise ValueError(f"{self.sample_rate} <= 0.")
