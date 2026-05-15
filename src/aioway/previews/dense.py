# Copyright (c) AIoWay Authors - All Rights Reserved

from torch import nn

from .previews import Preview


class Linear(Preview):
    NN = nn.Linear

    in_features: int
    "The number of features this layer can take in."

    out_features: int
    "The number of features this layer will put out."
