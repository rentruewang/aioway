# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.5
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %%
import tensordict as td
import torch

# %%
from aioway.torch import PrintTorchDisp, PrintTorchFunc, fake_mode

# %%
a = torch.randn(3, 4)
b = torch.randn(1, 1)

# %%
with PrintTorchDisp().activate(), PrintTorchFunc().activate():
    c = a + b

# %%
c

# %%
with fake_mode():
    a = torch.randn(3, 4)
    b = torch.randn(1, 1)

# %%
with PrintTorchDisp().activate(), PrintTorchFunc().activate():
    c = a + b

# %%
c

# %%
d = torch.randn_like(c)
d

# %% [markdown]
# ## TensorDicts

# %%
with PrintTorchDisp().activate(), PrintTorchFunc().activate():
    tdict = td.TensorDict({"c": c, "d": d})
    tdict

# %%
with PrintTorchDisp().activate(), PrintTorchFunc().activate():
    tdict.auto_batch_size_()
    tdict

# %%
with PrintTorchDisp().activate(), PrintTorchFunc().activate():
    tdict.auto_batch_size_()
    tdict

# %%
with PrintTorchDisp(rich=True).activate(), PrintTorchFunc().activate():
    double = tdict + tdict
    double

# %%
with PrintTorchDisp(rich=True).activate(), PrintTorchFunc().activate():
    double = tdict % tdict
    double

# %% [markdown]
# Seems like `tensordict` operations are not captured by `torch` functions (as `td.*` functions), as expected.
#
# It does translate `torch` calls to weird `_foreach_add` calls for `+`,
# but `%` seems unoptimized and calls `remainder` multiple times.

# %% [markdown]
# ## `torch.cond`

# %%
torch.cond

# %%
with PrintTorchDisp(rich=True).activate(), PrintTorchFunc().activate():
    r = torch.cond(1, lambda: a, lambda: b)

r

# %% [markdown]
# Nothing shows up!
