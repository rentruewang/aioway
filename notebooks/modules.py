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
from torch import nn

# %%
from aioway.fn import MFwdMode, MInitMode, module_fwd, module_init, set_nn_mode


# %%
class PrintModuleInit(MInitMode):
    def __call__(self, thunk):
        print(thunk)
        return thunk.do()


# %%
with PrintModuleInit().ctx():
    module_init(nn.Linear, 3, 5)
    module_init(nn.Dropout)


# %%
class PrintModuleForward(MFwdMode):
    def __call__(self, thunk):
        print(thunk)
        return thunk.do()


# %%
t = torch.randn(7, 3)

with PrintModuleInit().ctx(), PrintModuleForward().ctx():
    linear = module_init(nn.Linear, 3, 5)
    dropout = module_init(nn.Dropout)

    print("forward")

    t = module_fwd(linear, t)
    t = module_fwd(dropout, t)

# %%
t = torch.randn(7, 3)


with PrintModuleInit().ctx(), set_nn_mode(False, False), PrintModuleForward().ctx():
    linear = module_init(nn.Linear, 3, 5)
    dropout = module_init(nn.Dropout)

    print("forward")

    t = module_fwd(linear, t)
    t = module_fwd(dropout, t)

# %% [markdown]
# Note that outside contexts of `set_nn_mode` is disabled. This is consistent with how `set_torch_mode` works.

# %% [markdown]
# This means we copied torch modes' mechanism beautifully.
