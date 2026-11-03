"""
program.py: a program is a list of instructions, each running one entry of the catalog on named values:

    (outputs, op, inputs)        ("ex", shift, "corpus")   is   ex = shift(corpus)

op is a function (settings bound with functools.partial), the name of a value that holds one (a model), or
another program. A name can reach into a value: ex.input, path.end, batch.observation, states.-1.
Programs are values too: training takes one as its objective, loop repeats one, and `given` holds the
programs and constants a program uses. So each well-known method below is a short listing of catalog
entries, and printing it shows the group of every instruction. The entries are interfaces: the programs
print now, and run once the entries are implemented.

    python program.py      prints every program
"""

from __future__ import annotations

import operator
from collections.abc import Callable
from dataclasses import dataclass, field
from functools import partial
from typing import Any

from torch import nn

from catalog import (
    accuracy, autoencoder, blank, index, retrieve, discounted_return, dpo, droppable, dynamics, encoder, entry_of, env_tree, epsilon_greedy, fill, flatten,
    freeze, frozen_copy, given, greedy, group_relative, guided, imagined, independent, info_nce, labelled, lm_tree, log_prob, loop, lora,
    loss, mask, mcts, moe, most_confident, network, noise, patchify, planned, ppo_clip, q_target, quality_filter, reconstruction, reinforce,
    retrieval_accuracy, returns_to_go, sample, search_targets, selective_scan, shift, temperature, together, train, transfer, transitions,
    walk,
)
from data import Container, Seq

Instruction = tuple[str, Any, str]


def _get(env: dict[str, Any], name: str) -> Any:
    head, *rest = name.split(".")
    value = env[head]
    for part in rest:
        if part.lstrip("-").isdigit():
            value = value[int(part)]
        elif hasattr(value, part):
            value = getattr(value, part)
        else:
            value = (value.data if isinstance(value, Container) else value)[part]
    return value


@dataclass(frozen=True, eq=False)
class Program:
    """A list of instructions over named values; calling it runs them in order."""

    name: str
    inputs: str                       # the names it is called with, space-separated
    body: tuple[Instruction, ...]
    returns: str                      # the names it returns
    about: str = ""
    given: dict[str, Any] = field(default_factory=dict)

    def __call__(self, *args: Any) -> Any:
        names = self.inputs.split()
        if len(args) != len(names):
            raise TypeError(f"{self.name} takes ({', '.join(names)}), got {len(args)} values")
        env = {**self.given, **dict(zip(names, args))}
        for out, op, ins in self.body:
            f = _get(env, op) if isinstance(op, str) else op
            value = f(*(_get(env, n) for n in ins.split()))
            outs = out.split()
            env.update(zip(outs, value) if len(outs) > 1 else [(outs[0], value)])
        results = [_get(env, n) for n in self.returns.split()]
        return results[0] if len(results) == 1 else tuple(results)

    def __str__(self) -> str:
        calls = [(", ".join(out.split()), f"{_name(op)}({', '.join(ins.split())}{_settings(op)})", group_of(op)) for out, op, ins in self.body]
        w1, w2 = max(len(c[0]) for c in calls), max(len(c[1]) for c in calls)
        lines = [f"{self.name}({', '.join(self.inputs.split())}) → {self.returns}"] + ([f"  {self.about}"] if self.about else [])
        lines += [f"    {out:<{w1}} = {call:<{w2}}   {group}" for out, call, group in calls]
        for key, value in self.given.items():
            if isinstance(value, Program):
                lines += [f"  where {key} is"] + ["    " + line for line in str(value).splitlines()]
        return "\n".join(lines)


BUILDERS: set[Any] = set()  # functions that return a program (shown as "program")


def _name(op: Any) -> str:
    if isinstance(op, str):
        return op
    if isinstance(op, partial):
        return _name(op.func)
    return op.name if isinstance(op, Program) else getattr(op, "__name__", repr(op))


def _short(v: Any) -> str:
    return v.name if isinstance(v, Program) else getattr(v, "__name__", None) or repr(v)


def _settings(op: Any) -> str:
    if not isinstance(op, partial):
        return ""
    return "; " + ", ".join([*map(_short, op.args), *(f"{k}={_short(v)}" for k, v in op.keywords.items())])


def group_of(op: Any) -> str:
    """The catalog group of an instruction's op: "model" for a value called by name, "·" for plain helpers."""
    if isinstance(op, str):
        return "model"
    if isinstance(op, Program) or getattr(op, "func", op) in BUILDERS:
        return "program"
    e = entry_of(op)
    return e.group if e is not None else "·"


def groups(p: Program) -> list[str]:
    """The groups a program draws on, its sub-programs' included, in order of first use."""
    seen: list[str] = []
    for _, op, _ in p.body:
        g = group_of(op)
        if g not in seen and g not in ("model", "program", "·"):
            seen.append(g)
    for sub in p.given.values():
        if isinstance(sub, Program):
            seen += [g for g in groups(sub) if g not in seen]
    return seen


# ═══ Helpers: glue, not notions ═══════════════════════════════════════════════════════════════════


def towers(image: nn.Module, text: nn.Module) -> nn.ModuleDict:
    """Two encoders, trained as one module."""
    return nn.ModuleDict({"image": image, "text": text})


def head(name: str) -> bool:
    """Whether a parameter belongs to the readout."""
    return name.startswith("readout")


# ═══ Losses, as programs: a split, a model, a score ═══════════════════════════════════════════════

next_token = Program("next-token loss", "network text", (
    ("ex", shift, "text"),
    ("pred", "network", "ex.input"),
    ("loss", loss, "pred ex.target ex.weight"),
), "loss", "each token predicted from the ones before it")

masked_loss = Program("masked loss", "network text", (
    ("ex", mask, "text"),
    ("pred", "network", "ex.input"),
    ("loss", loss, "pred ex.target ex.weight"),
), "loss", "random holes, predicted")

supervised = Program("supervised loss", "network x y", (
    ("ex", labelled, "x y"),
    ("pred", "network", "ex.input"),
    ("loss", loss, "pred ex.target"),
), "loss", "the label as the target")

autoencoding = Program("reconstruction loss", "ae x", (
    ("ex", reconstruction, "x"),
    ("pred", "ae", "ex.input"),
    ("loss", loss, "pred ex.target"),
), "loss", "the input as its own target, through the bottleneck")

flow_matching = Program("conditional flow-matching loss", "velocity z labels", (
    ("ex", noise, "z"),
    ("pred", "velocity", "ex.input labels"),
    ("loss", loss, "pred ex.target"),
), "loss", "noised latents; the velocity back, now and then without the label")

contrastive = Program("contrastive loss", "towers images texts", (
    ("a", "towers.image", "images"),
    ("b", "towers.text", "texts"),
    ("loss", info_nce, "a b"),
), "loss", "which caption goes with which image, among the batch")

ppo_loss = Program("PPO loss", "agent batch", (
    ("logits", "agent.policy", "batch.observation"),
    ("value", "agent.critic", "batch.observation"),
    ("advantage", operator.sub, "batch.returns value"),
    ("log_p", log_prob, "logits batch.action"),
    ("policy_loss", ppo_clip, "log_p batch.log_prob advantage"),
    ("value_loss", loss, "value batch.returns"),
    ("loss", operator.add, "policy_loss value_loss"),
), "loss", "the clipped policy gradient with the critic as baseline, plus the critic's regression")

q_learning = Program("Q-learning loss", "q batch", (
    ("ex", q_target, "batch q"),
    ("pred", "q", "ex.input"),
    ("loss", loss, "pred ex.target ex.weight"),
), "loss", "the action taken learns toward r + γ·max Q(s′)")

dynamics_loss = Program("dynamics loss", "world batch", (
    ("ex", dynamics, "batch"),
    ("pred", "world", "ex.input"),
    ("loss", loss, "pred ex.target"),
), "loss", "from a state and an action, what follows")


def grpo(verify: Callable[..., Any], n: int = 8, length: int = 1) -> Program:
    return Program("GRPO loss", "policy prompts", (
        ("tree", lm_tree, "policy prompts"),
        ("choose", partial(independent, n), "sample"),
        ("path", partial(walk, steps=length), "tree choose"),
        ("reward", "verify", "path.end"),
        ("advantage", partial(group_relative, n=n), "reward"),
        ("loss", reinforce, "path advantage"),
    ), "loss", "n answers per prompt, verified, each pushed by how it did against its group", {"verify": verify, "sample": sample})


BUILDERS.add(grpo)

# ═══ Well-known work ══════════════════════════════════════════════════════════════════════════════


def language_model(name: str = "GPT", about: str = "a causal Transformer: shift, the loss, train; then walk the LM's tree",
                   build: Callable[..., Any] = network, steps: int = 200) -> Program:
    return Program(name, "corpus prompt", (
        ("ex", shift, "corpus"),
        ("network", build, "ex.input.known ex.target"),
        ("network", partial(train, steps=steps), "network next_token corpus"),
        ("tree", lm_tree, "network prompt"),
        ("path", partial(walk, choose=greedy, steps=8), "tree"),
    ), "path.end", about, {"next_token": next_token})


def mamba() -> Program:
    return language_model("Mamba", "GPT with a selective scan in place of attention", partial(network, mixer=selective_scan))


def mixtral() -> Program:
    return language_model("Mixtral / DeepSeek-V3", "GPT whose MLP is a routed mixture of experts", partial(network, mlp=moe))


def vit(steps: int = 100) -> Program:
    return Program("ViT", "images labels", (
        ("patches", partial(patchify, size=4), "images"),
        ("tokens", flatten, "patches"),
        ("network", network, "tokens labels"),
        ("network", partial(train, steps=steps), "network supervised tokens labels"),
        ("pred", "network", "tokens"),
        ("acc", accuracy, "pred labels"),
    ), "acc", "an image as a sequence of patches, read by a Transformer", {"supervised": supervised})


def clip(dim: int = 16, steps: int = 150) -> Program:
    return Program("CLIP", "images texts", (
        ("patches", partial(patchify, size=4), "images"),
        ("image", partial(encoder, dim=dim), "patches"),
        ("text", partial(encoder, dim=dim), "texts"),
        ("towers", towers, "image text"),
        ("towers", partial(train, steps=steps), "towers contrastive patches texts"),
        ("a", "towers.image", "patches"),
        ("b", "towers.text", "texts"),
        ("acc", retrieval_accuracy, "a b"),
    ), "acc", "an image encoder and a text encoder, trained so that each image picks out its caption", {"contrastive": contrastive})


def stable_diffusion(latent: int = 4, steps: int = 150, w: float = 3.0) -> Program:
    return Program("Stable Diffusion", "images labels", (
        ("ae", partial(autoencoder, latent=latent), "images"),
        ("ae", partial(train, steps=steps), "ae autoencoding images"),
        ("ae", frozen_copy, "ae"),
        ("z", "ae.encode", "images"),
        ("ex", noise, "z"),
        ("velocity", network, "ex.input.known ex.target"),
        ("velocity", partial(droppable, p=0.1), "velocity"),
        ("velocity", partial(train, steps=steps), "velocity flow_matching z labels"),
        ("m_c", given, "velocity labels"),
        ("m_u", given, "velocity nothing"),
        ("m", partial(guided, w=w), "m_c m_u"),
        ("start", blank, "z"),
        ("latents", partial(fill, schedule=together, steps=10), "start m"),
        ("images", "ae.decode", "latents"),
    ), "images", "flow matching in an autoencoder's latent space, with a label, sampled with classifier-free guidance",
        {"autoencoding": autoencoding, "flow_matching": flow_matching, "nothing": None})


def lora_finetune(rank: int = 4, steps: int = 150) -> Program:
    return Program("LoRA finetuning", "network corpus prompt", (
        ("network", partial(lora, rank=rank), "network"),
        ("network", partial(train, steps=steps), "network next_token corpus"),
        ("tree", lm_tree, "network prompt"),
        ("path", partial(walk, choose=greedy, steps=8), "tree"),
    ), "path.end", "a pretrained model, frozen, with a trainable low-rank update", {"next_token": next_token})


def dpo_program(beta: float = 0.1, steps: int = 60) -> Program:
    return Program("DPO", "policy chosen rejected", (
        ("reference", frozen_copy, "policy"),
        ("objective", partial(dpo, beta=beta), "reference"),
        ("policy", partial(train, steps=steps), "policy objective chosen rejected"),
    ), "policy", "preference pairs as a classification of log-ratios against a frozen reference")


def alphazero(rounds: int = 8, simulations: int = 32, steps: int = 30, length: int = 16) -> Program:
    round_ = Program("self-play round", "network game", (
        ("tree", "game", "network"),
        ("search", partial(mcts, simulations=simulations, two_player=True), "tree"),
        ("path", partial(walk, choose=sample, steps=length), "search"),
        ("ex", search_targets, "path search"),
        ("network", partial(train, steps=steps), "network supervised ex.input.known ex.target"),
    ), "network", "play yourself with search; learn the search's choices and each game's outcome", {"supervised": supervised})
    return Program("AlphaZero", "network game", (
        ("network", partial(loop, times=rounds), "round network game"),
        ("tree", "game", "network"),
        ("path", partial(walk, choose=greedy, steps=length), "tree"),
    ), "path", "a policy-value network that amortizes MCTS, trained by self-play", {"round": round_})


def r1(n: int = 8, length: int = 1, steps: int = 100) -> Program:
    return Program("DeepSeek-R1 (GRPO)", "policy prompts verify", (
        ("objective", partial(grpo, n=n, length=length), "verify"),
        ("policy", partial(train, steps=steps), "policy objective prompts"),
        ("tree", lm_tree, "policy prompts"),
        ("path", partial(walk, choose=greedy, steps=length), "tree"),
        ("reward", "verify", "path.end"),
    ), "reward", "RL on checkable answers: walk, verify, group-relative advantages, REINFORCE")


def diffusion_lm(steps: int = 600) -> Program:
    return Program("Diffusion LLM", "corpus", (
        ("ex", mask, "corpus"),
        ("network", network, "ex.input.known ex.target"),
        ("network", partial(train, steps=steps), "network masked_loss corpus"),
        ("start", blank, "corpus"),
        ("texts", partial(fill, schedule=most_confident, steps=8), "start network"),
    ), "texts", "masked prediction at every ratio; generation unmasks the surest holes first", {"masked_loss": masked_loss})


# ═══ Compositions the catalog points to ═══════════════════════════════════════════════════════════


def pretrain_finetune(steps: int = 150) -> Program:
    return Program("Pretrain → finetune", "corpus texts labels", (
        ("ex", shift, "corpus"),
        ("lm", network, "ex.input.known ex.target"),
        ("lm", partial(train, steps=steps), "lm next_token corpus"),
        ("classifier", network, "texts labels"),
        ("classifier", transfer, "lm classifier"),
        ("classifier", partial(freeze, trainable=head), "classifier"),
        ("classifier", partial(train, steps=steps), "classifier supervised texts labels"),
    ), "classifier", "learn from raw text, then reuse the body under a new head", {"next_token": next_token, "supervised": supervised})


def ppo(rounds: int = 30, horizon: int = 20, epochs: int = 4) -> Program:
    round_ = Program("PPO round", "agent env", (
        ("tree", env_tree, "env agent.policy"),
        ("path", partial(walk, choose=sample, steps=horizon), "tree"),
        ("returns", partial(returns_to_go, gamma=0.9), "path.rewards"),
        ("batch", transitions, "path returns"),
        ("agent", partial(train, steps=epochs), "agent ppo_loss batch"),
    ), "agent", "walk the environment, then a few clipped steps on what was seen", {"ppo_loss": ppo_loss})
    return Program("PPO (actor-critic)", "agent env", (
        ("agent", partial(loop, times=rounds), "round agent env"),
        ("tree", env_tree, "env agent.policy"),
        ("path", partial(walk, choose=greedy, steps=horizon), "tree"),
        ("score", discounted_return, "path"),
    ), "score", "a policy and a critic, improved round by round on their own walks", {"round": round_})


def dqn(rounds: int = 30, horizon: int = 20, steps: int = 10, eps: float = 0.3) -> Program:
    round_ = Program("DQN round", "q env", (
        ("tree", env_tree, "env q"),
        ("tree", partial(epsilon_greedy, eps=eps), "tree"),
        ("path", partial(walk, choose=sample, steps=horizon), "tree"),
        ("batch", transitions, "path"),
        ("q", partial(train, steps=steps), "q q_learning batch"),
    ), "q", "walk ε-greedily on the Q-values, then regress them toward their targets", {"q_learning": q_learning})
    return Program("DQN", "q env", (
        ("q", partial(loop, times=rounds), "round q env"),
        ("tree", env_tree, "env q"),
        ("path", partial(walk, choose=greedy, steps=horizon), "tree"),
        ("score", discounted_return, "path"),
    ), "score", "values learned by bootstrapping, acted on greedily", {"round": round_})


def world_model_planning(horizon: int = 20, steps: int = 300) -> Program:
    return Program("Planning in a learned world model", "env", (
        ("tree", env_tree, "env"),
        ("path", partial(walk, choose=sample, steps=horizon), "tree"),
        ("batch", transitions, "path"),
        ("ex", dynamics, "batch"),
        ("world", network, "ex.input.known ex.target"),
        ("world", partial(train, steps=steps), "world dynamics_loss batch"),
        ("dream", imagined, "tree world"),
        ("plan", partial(planned, horizon=8, samples=32), "tree dream"),
        ("path", partial(walk, choose=greedy, steps=horizon), "plan"),
        ("score", discounted_return, "path"),
    ), "score", "explore at random, learn what follows each action, act by planning inside that model (Dreamer, MuZero)",
        {"dynamics_loss": dynamics_loss})


def synthetic_data(n: int = 8, length: int = 4, keep: float = 0.25) -> Program:
    return Program("Synthetic data", "generator prompts verify", (
        ("tree", lm_tree, "generator prompts"),
        ("tree", partial(temperature, t=1.5), "tree"),
        ("choose", partial(independent, n), "sample"),
        ("path", partial(walk, steps=length), "tree choose"),
        ("samples", Seq, "path.end"),
        ("scores", "verify", "path.end"),
        ("kept", partial(quality_filter, keep=keep), "samples scores"),
    ), "kept", "sample a model's tree widely, keep what a verifier scores best", {"sample": sample})


def rag(k: int = 4) -> Program:
    return Program("Retrieval-augmented generation", "lm embed documents question", (
        ("vectors", "embed", "documents"),
        ("memory", partial(index, kind="hnsw"), "vectors"),
        ("prompt", partial(retrieve, k=k), "memory documents question embed"),
        ("tree", lm_tree, "lm prompt"),
        ("path", partial(walk, choose=greedy, steps=64), "tree"),
    ), "path.end", "index the documents' embeddings, put the nearest ones in front of the question, decode")


PROGRAMS: list[Program] = [
    language_model(), vit(), clip(), stable_diffusion(), mamba(), mixtral(), lora_finetune(), dpo_program(), alphazero(), r1(), diffusion_lm(),
    pretrain_finetune(), ppo(), dqn(), world_model_planning(), synthetic_data(), rag(),
]

if __name__ == "__main__":
    for p in PROGRAMS:
        print(p, end="\n\n")
