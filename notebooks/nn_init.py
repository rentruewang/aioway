# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.3
#   kernelspec:
#     display_name: .venv
#     language: python
#     name: python3
# ---

# %% [markdown]
# Mess is a layer between `nn.Module` and the compiler of `aioway`,
# it sits there providing information as to what flags to expect, and what signature to pass.

# %%
from torch import nn

from aioway._modes import NnInitFn

# %%
from aioway.hop import find_nn_init

# %%
nn_init = find_nn_init(NnInitFn(func=nn.Linear, in_features=3, out_features=5))
nn_init

# %%
assert nn_init
module = nn_init()
module

# %% [markdown]
# Note that our `nn_init` has the same signature as `module`, and almost the same `repr` as well.
