# Copyright (c) AIoWay Authors - All Rights Reserved

import typing


class Domain(typing.Protocol):
    """
    The base type for domains.

    Domains are the core data representation in `aioway`,
    describing how a batch of data behaves.
    """
