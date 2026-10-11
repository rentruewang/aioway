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

# %%
import torch

# %%
from rich import pretty

# %%
from aioway.ir import FuncProgram, TorchFuncDag, render_stateless_program
from aioway.t import fake_mode

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
tracer = TorchFuncDag()

with fake_mode():
    x, y = torch.zeros(3), torch.zeros(3)

    with tracer.activate():
        summed = torch.add(x, y)
        product = torch.mul(summed, x)
        difference = torch.sub(product, summed)
        activated = torch.relu(difference)

iset = FuncProgram.from_instr_list(tracer.thunks)

# %%
render_stateless_program(iset, "example_program", "x", "y")
