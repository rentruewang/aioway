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
from aioway.tags import SampleRateTag

# %%
t = torch.randn(3, 4)

# %%
sr = SampleRateTag(t, 100)

# %%
s = torch.randn(4, 5)

# %%
sr.attach(s)

# %%
SampleRateTag.extract(t)

# %%
SampleRateTag.extract(s)

# %%
SampleRateTag.extract(t) == SampleRateTag.extract(s)

# %%
t.__aioway_audio_sample_rate__
