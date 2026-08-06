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

# %% [markdown]
#
# # Introduction to TorchRL
#
# Get started with reinforcement learning in PyTorch.
#

# %% [markdown]
# TorchRL is an open-source Reinforcement Learning (RL) library for PyTorch.
# This tutorial provides a hands-on introduction to its main components.
#
# **Key features:**
#
# - **PyTorch-native**: Seamless integration with PyTorch's ecosystem
# - **Modular**: Easily swap components and build custom pipelines
# - **Efficient**: Optimized for both research and production
# - **Comprehensive**: Environments, modules, losses, collectors, and more
#
# By the end of this tutorial, you'll understand how TorchRL's components
# work together to build RL training pipelines. Let's start with a quick
# example to see what's possible:
#
#

# %% [markdown]
# ## Quick Start
#
# Before diving into the details, here's a taste of what TorchRL can do.
# In just a few lines, we can create an environment, build a policy, and
# collect a trajectory:
#
#
#
# %%
import tensordict as td
import torch
from tensordict import nn as tnn
from torch import nn
from torchrl import collectors as trl_cols
from torchrl import data as trl_data
from torchrl import envs as trl_envs
from torchrl import modules as trl_mods
from torchrl import objectives as trl_objs

env = trl_envs.GymEnv("CartPole-v1")
actor = trl_mods.QValueActor(
    trl_mods.MLP(
        in_features=env.observation_spec["observation"].shape[-1],
        out_features=2,
        num_cells=[64, 64],
    ),
    in_keys=["observation"],
    spec=env.action_spec,
)
rollout = env.rollout(max_steps=200, policy=actor)
print(
    f"Collected {rollout.shape[0]} steps, total reward: {rollout['next', 'reward'].sum().item():.0f}"
)

# %% [markdown]
# That's it! We wrapped a Gym environment, created a Q-value actor with an
# trl_mods.MLP backbone, and used :meth:`~torchrl.envs.EnvBase.rollout` to collect
# a full trajectory. The result is a :class:`~td.TensorDict`
# containing observations, actions, rewards, and more.
#
# Now let's understand each component in detail.
#
# ## td.TensorDict: The Data Backbone
#
# At the heart of TorchRL is :class:`~td.TensorDict` - a dictionary-like
# container that holds tensors and supports batched operations. Think of it as
# a "tensor of dictionaries" or a "dictionary of tensors" that knows about its
# batch dimensions.
#
# Why td.TensorDict? In RL, we constantly pass around groups of related tensors:
# observations, actions, rewards, done flags, next observations, etc. td.TensorDict
# keeps these organized and lets us manipulate them as a unit.
#
#

# %%
# Create a td.TensorDict representing a batch of 4 transitions
batch_size = 4
data = td.TensorDict(
    obs=torch.randn(batch_size, 3),
    action=torch.randn(batch_size, 2),
    reward=torch.randn(batch_size, 1),
    batch_size=[batch_size],
)
print(data)

# %% [markdown]
# TensorDicts support all the operations you'd expect from PyTorch tensors.
# You can index them, slice them, move them between devices, and stack them
# together - all while keeping the dictionary structure intact:
#
#

# %%
# Indexing works just like tensors - grab the first transition
print("First element:", data[0])
print("Slice:", data[:2])

# Device transfer moves all contained tensors
data_cpu = data.to("cpu")

# Stacking is especially useful for building trajectories
data2 = data.clone()
stacked = torch.stack([data, data2], dim=0)
print("Stacked shape:", stacked.batch_size)

# %% [markdown]
# TensorDicts can also be nested, which is useful for organizing complex
# observations (e.g., an agent that receives both image pixels and vector
# state) or for separating "current" from "next" step data:
#
#

# %%
nested = td.TensorDict(
    observation=td.TensorDict(
        pixels=torch.randn(4, 3, 84, 84),
        vector=torch.randn(4, 10),
        batch_size=[4],
    ),
    action=torch.randn(4, 2),
    batch_size=[4],
)
print(nested)

# %% [markdown]
# ## Environments
#
# TorchRL provides a unified interface for RL environments. Whether you're
# using Gym, DMControl, IsaacGym, or other simulators, the API stays the same:
# environments accept and return TensorDicts.
#
# **Creating Environments**
#
# The simplest way to create an environment is with :class:`~torchrl.envs.trl_envs.GymEnv`,
# which wraps any Gymnasium (or legacy Gym) environment:
#
#

# %%
env = trl_envs.GymEnv("Pendulum-v1")
print("Action spec:", env.action_spec)
print("Observation spec:", env.observation_spec)

# %% [markdown]
# Every environment has *specs* that describe the shape and bounds of
# observations, actions, rewards, and done flags. These specs are essential
# for building correctly-shaped networks and for validating data.
#
# The environment interaction follows a familiar pattern - reset, then step:
#
#

# %%
tdict = env.reset()
print("Reset output:", tdict)

# Sample a random action and take a step
tdict["action"] = env.action_spec.rand()
tdict = env.step(tdict)
print("Step output:", tdict)

# %% [markdown]
# Notice that :meth:`~torchrl.envs.EnvBase.step` returns the same td.TensorDict
# with additional keys filled in: the ``"next"`` sub-td.TensorDict contains the
# resulting observation, reward, and done flag.
#
# **Transforms**
#
# Just like torchvision transforms for images, TorchRL provides transforms
# for environments. These modify observations, actions, or rewards in a
# composable way. Common uses include normalizing observations, stacking
# frames, or adding step counters:
#
#

# %%

env = trl_envs.TransformedEnv(
    trl_envs.GymEnv("Pendulum-v1"),
    trl_envs.Compose([trl_envs.StepCounter(max_steps=200)]),
)
print("Transformed env:", env)

# %% [markdown]
# **Batched Environments**
#
# RL algorithms are data-hungry. Running multiple environment instances in
# parallel can dramatically speed up data collection. TorchRL's
# :class:`~trl_envs.SerialEnv` runs environments multiple times,
# returning batched TensorDicts:
#

# %%


def make_env():
    return trl_envs.GymEnv("Pendulum-v1")


# Run 4 environments in parallel
vec_env = trl_envs.SerialEnv(4, make_env)
tdict = vec_env.reset()
print("Batched reset:", tdict.batch_size)

tdict["action"] = vec_env.action_spec.rand()
tdict = vec_env.step(tdict)
print("Batched step:", tdict.batch_size)

vec_env.close()

# %% [markdown]
# The batch dimension (4 in this case) propagates through all tensors,
# making it easy to process multiple environments with a single forward pass.
#
# ## Modules and Policies
#
# TorchRL extends PyTorch's ``nn.Module`` system with modules that read from
# and write to TensorDicts. This makes it easy to build policies that
# integrate seamlessly with the environment interface.
#
# **tnn.TensorDictModule**
#
# The core building block is :class:`~tensordict.nn.tnn.TensorDictModule`. It wraps
# any ``nn.Module`` and specifies which td.TensorDict keys to read as inputs and
# which keys to write as outputs:
#
#

# %%
module = nn.Linear(3, 2)
td_module = tnn.TensorDictModule(module, in_keys=["observation"], out_keys=["action"])

# The module reads "observation" and writes "action"
t = td.TensorDict(observation=torch.randn(4, 3), batch_size=[4])
td_module(t)
print(t)

# %% [markdown]
# This pattern has a powerful benefit: modules become composable. You can
# chain them together, and each module only needs to know about its own
# input/output keys.
#
# **Built-in Networks**
#
# TorchRL includes common network architectures used in RL. These are
# regular PyTorch modules that you can wrap with tnn.TensorDictModule:
#
#

# %%

# trl_mods.MLP for vector observations - specify input/output dims and hidden layers
mlp = trl_mods.MLP(in_features=64, out_features=10, num_cells=[128, 128])
print(mlp(torch.randn(4, 64)).shape)

# trl_mods.ConvNet for image observations - outputs a flat feature vector
cnn = trl_mods.ConvNet(num_cells=[32, 64], kernel_sizes=[8, 4], strides=[4, 2])
print(cnn(torch.randn(4, 3, 84, 84)).shape)

# %% [markdown]
# **Probabilistic Policies**
#
# Many RL algorithms (PPO, SAC, etc.) use stochastic policies that output
# probability distributions over actions. TorchRL provides
# :class:`~tensordict.nn.tnn.ProbabilisticTensorDictModule` to sample from
# distributions and optionally compute log-probabilities:
#
#

# %%
# The network outputs mean and std (via trl_mods.NormalParamExtractor)
net = nn.Sequential(
    nn.Linear(3, 64), nn.ReLU(), nn.Linear(64, 4), trl_mods.NormalParamExtractor()
)
backbone = tnn.TensorDictModule(net, in_keys=["observation"], out_keys=["loc", "scale"])

# Combine backbone with a distribution sampler
policy = tnn.ProbabilisticTensorDictSequential(
    backbone,
    tnn.ProbabilisticTensorDictModule(
        in_keys=["loc", "scale"],
        out_keys=["action"],
        distribution_class=trl_mods.TanhNormal,
        return_log_prob=True,
    ),
)

tdict = td.TensorDict(observation=torch.randn(4, 3), batch_size=[4])
policy(tdict)
print("Sampled action:", tdict["action"].shape)
print("Log prob:", tdict["action_log_prob"].shape)

# %% [markdown]
# The ``trl_mods.TanhNormal`` distribution squashes samples to [-1, 1], which is useful
# for continuous control. The log-probability accounts for this transformation,
# which is crucial for policy gradient methods.
#
# ## Data Collection
#
# In RL, we need to repeatedly collect experience from the environment.
# While you can write your own rollout loop, TorchRL's *collectors* handle
# this efficiently, including batching, device management, and multi-process
# collection.
#
# The :class:`~trl_cols.Collector` collects data
# synchronously - it waits for a batch to be ready before returning:
#
#

# %%

# A simple deterministic policy for demonstration
actor = tnn.TensorDictModule(
    nn.Linear(3, 1), in_keys=["observation"], out_keys=["action"]
)

collector = trl_cols.Collector(
    create_env_fn=lambda: trl_envs.GymEnv("Pendulum-v1"),
    policy=actor,
    frames_per_batch=200,  # Collect 200 frames per iteration
    total_frames=1000,  # Stop after 1000 total frames
)

for batch in collector:
    print(
        f"Collected batch: {batch.shape}, reward: {batch['next', 'reward'].mean():.2f}"
    )

collector.shutdown()

# %% [markdown]
# For async collection (useful when training takes longer than collecting),
# see :class:`~torchrl.collectors.MultiAsyncCollector`.
#
# ## Replay Buffers
#
# Most RL algorithms don't learn from experience immediately - they store
# transitions in a buffer and sample mini-batches for training. TorchRL's
# replay buffers handle this efficiently:
#
#

# %%

buffer = trl_data.ReplayBuffer(storage=trl_data.LazyTensorStorage(max_size=10000))

# Add a batch of experience
buffer.extend(
    td.TensorDict(obs=torch.randn(100, 4), action=torch.randn(100, 2), batch_size=[100])
)

# Sample a mini-batch for training
sample = buffer.sample(32)
print("Sampled batch:", sample.batch_size)

# %% [markdown]
# The :class:`~torchrl.data.replay_buffers.trl_data.LazyTensorStorage` allocates memory lazily based
# on the first batch added. For prioritized experience replay (used in DQN
# variants), use :class:`~trl_data.PrioritizedReplayBuffer`:
#
#

# %%

buffer = trl_data.PrioritizedReplayBuffer(
    alpha=0.6,  # Priority exponent
    beta=0.4,  # Importance sampling exponent
    storage=trl_data.LazyTensorStorage(max_size=10000),
)
buffer.extend(td.TensorDict(obs=torch.randn(100, 4), batch_size=[100]))

# Use return_info=True to get sampling metadata (indices, weights)
sample, info = buffer.sample(32, return_info=True)
print("Prioritized sample indices:", info["index"][:5], "...")  # First 5 indices

# %% [markdown]
# ## Loss Functions
#
# The final piece is the objective function. TorchRL provides loss classes
# for major RL algorithms, encapsulating the often-complex loss computations:
#
# - :class:`~torchrl.objectives.DQNLoss` - Deep Q-Networks
# - :class:`~torchrl.objectives.DDPGLoss` - Deep Deterministic Policy Gradient
# - :class:`~torchrl.objectives.SACLoss` - Soft Actor-Critic
# - :class:`~torchrl.objectives.PPOLoss` - Proximal Policy Optimization
# - :class:`~torchrl.objectives.TD3Loss` - Twin Delayed DDPG
#
# Here's how to set up a DQN loss. We create a Q-network wrapped in a
# :class:`~torchrl.modules.trl_mods.QValueActor`, which handles action selection:
#
#

# %%

qnet = tnn.TensorDictModule(
    nn.Sequential(nn.Linear(4, 64), nn.ReLU(), nn.Linear(64, 2)),
    in_keys=["observation"],
    out_keys=["action_value"],
)

# trl_mods.QValueActor wraps the Q-network to select actions and output chosen values

actor = trl_mods.QValueActor(
    qnet, in_keys=["observation"], spec=trl_data.Categorical(n=2)
)
loss_fn = trl_objs.DQNLoss(actor, action_space="categorical")

# %% [markdown]
# The loss function expects batches with specific keys. Let's create a
# dummy batch to see it in action:
#
#

# %%
batch = td.TensorDict(
    observation=torch.randn(32, 4),
    action=torch.randint(0, 2, (32,)),
    next=td.TensorDict(
        observation=torch.randn(32, 4),
        reward=torch.randn(32, 1),
        done=torch.zeros(32, 1, dtype=torch.bool),
        terminated=torch.zeros(32, 1, dtype=torch.bool),
        batch_size=[32],
    ),
    batch_size=[32],
)

loss_td = loss_fn(batch)
print("Loss:", loss_td["loss"])

# %% [markdown]
# The loss function handles target network updates, Bellman backup
# computation, and all the bookkeeping needed for stable training.
#
# ## Putting It All Together
#
# Now let's see how all these components work together in a complete
# training loop. We'll train a simple DQN agent on CartPole:
#
#

# %%
torch.manual_seed(0)

# 1. Create the environment
env = trl_envs.GymEnv("CartPole-v1")

# 2. Build a Q-network and wrap it as a policy
qnet = tnn.TensorDictModule(
    nn.Sequential(nn.Linear(4, 128), nn.ReLU(), nn.Linear(128, 2)),
    in_keys=["observation"],
    out_keys=["action_value"],
)
policy = trl_mods.QValueActor(qnet, in_keys=["observation"], spec=env.action_spec)

# 3. Set up the data collector
collector = trl_cols.Collector(
    create_env_fn=lambda: trl_envs.GymEnv("CartPole-v1"),
    policy=policy,
    frames_per_batch=100,
    total_frames=2000,
)

# 4. Create a replay buffer
buffer = trl_data.ReplayBuffer(storage=trl_data.LazyTensorStorage(max_size=10000))

# 5. Set up the loss and optimizer (pass the trl_mods.QValueActor, not just the network)
loss_fn = trl_objs.DQNLoss(policy, action_space=env.action_spec)
optimizer = torch.optim.Adam(policy.parameters(), lr=1e-3)

# 6. Training loop: collect -> store -> sample -> train
for i, batch in enumerate(collector):
    # Store collected experience
    buffer.extend(batch)

    # Wait until we have enough data
    if len(buffer) < 100:
        continue

    # Sample a batch and compute the loss
    sample = buffer.sample(64)
    loss = loss_fn(sample)

    # Standard PyTorch optimization step
    optimizer.zero_grad()
    loss["loss"].backward()
    optimizer.step()

    if i % 5 == 0:
        print(f"Step {i}: loss={loss['loss'].item():.3f}")

collector.shutdown()
env.close()

# %% [markdown]
# This is a minimal example - a production DQN would include target network
# updates, epsilon-greedy exploration, and more. Check out the full
# implementations in ``sota-implementations/dqn/``.
#
# ## What's Next?
#
# This tutorial covered the basics. TorchRL has much more to offer:
#
# **Tutorials:**
#
# - [PPO Tutorial](../tutorials/coding_ppo.html) - Train PPO on MuJoCo
# - [DQN Tutorial](../tutorials/coding_dqn.html) - Deep Q-Learning from scratch
# - [Multi-Agent RL](../tutorials/multiagent_ppo.html) - Cooperative and competitive agents
#
# **SOTA Implementations:**
#
# The [sota-implementations/](https://github.com/pytorch/rl/tree/main/sota-implementations)
# folder contains production-ready implementations of:
#
# - PPO, A2C, SAC, TD3, DDPG, DQN
# - Offline RL: CQL, IQL, Decision Transformer
# - Multi-agent: IPPO, QMIX, MADDPG
# - LLM training: GRPO, Expert Iteration
#
# **Advanced Features:**
#
# - Distributed training with Ray and RPC
# - Offline RL datasets (D4RL, Minari)
# - Model-based RL (Dreamer)
# - LLM integration for RLHF
#
# **Resources:**
#
# - [API Reference](https://pytorch.org/rl/reference/index.html)
# - [GitHub](https://github.com/pytorch/rl)
# - [Contributing Guide](https://github.com/pytorch/rl/blob/main/CONTRIBUTING.md)
#
#
#
