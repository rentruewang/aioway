# Copyright (c) AIoWay Authors - All Rights Reserved

import collections.abc as cabc
import dataclasses as dcls

import loguru as L

from aioway.io import Dset

__all__ = ["DsetMgr"]

# Dataset manager ====


@dcls.dataclass
class DsetMgr(cabc.MutableMapping[str, Dset]):
    """
    The dataset manager, a `dict[str, Dset]` with CRUD.

    Like tables in SQL: any number of datasets can be read at once,
    but only 1 can be written to at a time.
    """

    dsets: dict[str, Dset] = dcls.field(default_factory=dict)
    """
    The datasets, by name.
    """

    def __getitem__(self, key: str) -> Dset:
        return self.dsets[key]

    def __setitem__(self, key: str, value: Dset) -> None:
        self.dsets[key] = value

    def __delitem__(self, key: str) -> None:
        del self.dsets[key]

    def __iter__(self) -> cabc.Iterator[str]:
        return iter(self.dsets)

    def __len__(self) -> int:
        return len(self.dsets)
