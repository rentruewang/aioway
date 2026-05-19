# Copyright (c) AIoWay Authors - All Rights Reserved


import tensordict as td

__all__ = ["tdict_rename", "tdict_all_equal"]


def tdict_rename(tdict: td.TensorDict, **renames: str):
    return td.TensorDict({renames.get(key, key): value for key, value in tdict.items()})


def tdict_all_equal(left: td.TensorDict, right: td.TensorDict, /):
    if left.keys() != right.keys():
        return False

    eq: td.TensorDict = left == right
    return eq.all()
