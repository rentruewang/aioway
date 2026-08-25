# Copyright (c) AIoWay Authors - All Rights Reserved

from sklearn import decomposition

from aioway._utils import FloatArray
from aioway.trainers import Step

__all__ = ["PcaStep"]


class PcaStep(Step[FloatArray]):
    pca: decomposition.IncrementalPCA
    data: FloatArray

    def __call__(self) -> FloatArray:
        self.pca.partial_fit(self.data)
        return self.pca.transform(self.data)
