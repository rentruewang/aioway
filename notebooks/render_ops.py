# ---
# jupyter:
#   jupytext:
#     formats: py:percent
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

import rich

# %%
import torch
from rich import pretty

from aioway._signs import Param

# %%
from aioway.ops import CallOp, FuncInputOp, OpFunc

pretty.install()

# %% [markdown]
# The following cell will have the following function:
# note that only (x, y) are the inputs.

# %% [markdown]
# step 0: summed = add(x, y)
# step 1: product = mul(summed, x)
# step 2: difference = sub(product, summed)
# step 3: activated = relu(difference)

# %%
x = FuncInputOp(param=Param.build("x"), typ=torch.Tensor)
y = FuncInputOp(param=Param.build("y"), typ=torch.Tensor)

summed = CallOp(func=torch.add, args=(x, y), kwargs={})
product = CallOp(func=torch.mul, args=(summed, x), kwargs={})
difference = CallOp(func=torch.sub, args=(product, summed), kwargs={})
activated = CallOp(func=torch.relu, args=(difference,), kwargs={})

func = OpFunc(activated)

# %%
func

# %%
rich.print(func)

# %%
print(func)

# %% [markdown]
# It is also callable.

# %%
func(torch.ones(3), torch.full((3,), 2.0))

# %% [markdown]
# Modules are rendered as `self.<name>`, and the whole thing as an `nn.Module`.
#
# step 0: hidden = linear(x)
# step 1: activated = relu(hidden)
# step 2: summed = add(activated, y)
# step 3: normed = norm(summed)
# step 4: out = linear(normed)  <- same module again, same `self.linear_1`

# %%
from torch import nn

from aioway.ops import ModuleThunkOp
from aioway.ops.codes import render_op_module

linear = nn.Linear(3, 3)
norm = nn.LayerNorm(3)

x = FuncInputOp(param=Param.build("x"), typ=torch.Tensor)
y = FuncInputOp(param=Param.build("y"), typ=torch.Tensor)

hidden = ModuleThunkOp(func=linear, args=(x,), kwargs={})
activated = CallOp(func=torch.relu, args=(hidden,), kwargs={})
summed = CallOp(func=torch.add, args=(activated, y), kwargs={})
normed = ModuleThunkOp(func=norm, args=(summed,), kwargs={})
out = ModuleThunkOp(func=linear, args=(normed,), kwargs={})

module_func = OpFunc(out)

# %%
rich.print(render_op_module(module_func, name="Example"))

# %%
print(render_op_module(module_func, name="Example"))

# %%
module_func(torch.ones(3), torch.full((3,), 2.0))
