# ---
# jupyter:
#   jupytext:
#     formats: py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.1
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %%
import torch

# %%
from aioway.fn import PrintTorDis, PrintTorFunc, mode_off

# %%
a = torch.randn(3, 4)
b = torch.randn(1, 1)

# %%
with PrintTorDis().ctx(), PrintTorFunc().ctx():
    c = a + b

# %% [markdown]
# There is a bunch of output because that is desired.

# %%
c

# %%
with PrintTorDis().ctx(), PrintTorFunc().ctx(), mode_off():
    c = a + b

# %% [markdown]
# Have nothing at all! This is because `torch_mode_off` disable things "outside" its scope.

# %%
with PrintTorDis().ctx(), mode_off(), PrintTorFunc().ctx():
    c = a + b

# %% [markdown]
# Only "function::" calls are left. `torch_mode_off` disabled all the contexts entered before it.
