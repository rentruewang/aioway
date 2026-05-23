# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.3
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %%
import torch

# %%
from aioway.specs import SampleRateTag

# %%
t = torch.randn(3, 4)

# %%
sr = SampleRateTag(100)
sr.attach(t)

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
