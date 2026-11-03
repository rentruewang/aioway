"""
program.py: a program is an expression tree over the entries of the catalog.

    corpus, prompt = inputs("corpus prompt")
    ex = call(shift, corpus).named("ex")                       # a call: an entry applied to other nodes
    net = call(network, ex.input.known, ex.target)             # .attr reaches into a node's value
    out = net(prompt)                                          # calling a node calls the value it stands for

Leaves are inputs and constants (settings, other programs). A node used twice is one object, so the tree is a
DAG: it runs once and prints once. Programs are values: training takes one as its objective, loop repeats one.

Modifying a program is rewriting its tree, which returns a new program (the old one is unchanged):

    GPT.rewrite(calls(walk), lambda n: n.with_settings(steps=32))            # change one node's settings
    GPT.rewrite(calls(network), lambda n: call(network, *n.args, mixer=selective_scan))   # swap a subtree
    GPT.find(calls(train))                                                   # find nodes

Printing shows the tree from its output down, each call with its catalog group. The entries are interfaces: the
programs print and rewrite now, and run once the entries are implemented.

    python program.py      prints every program
"""

from __future__ import annotations

import operator
from collections.abc import Callable
from dataclasses import dataclass, replace
from functools import partial
from typing import Any

from torch import nn

from catalog import (
    accuracy, autoencoder, blank, discounted_return, dpo, droppable, dynamics, encoder, entry_of, env_tree, epsilon_greedy, fill, flatten,
    freeze, frozen_copy, given, greedy, group_relative, guided, imagined, independent, index, info_nce, labelled, lm_tree, log_prob, loop,
    lora, loss, mask, mcts, moe, most_confident, network, noise, patchify, planned, ppo_clip, q_target, quality_filter, reconstruction,
    reinforce, retrieval_accuracy, retrieve, returns_to_go, sample, search_targets, selective_scan, shift, temperature, together, train,
    transfer, transitions, walk,
)
from data import Container, Seq

# ═══ The tree ═════════════════════════════════════════════════════════════════════════════════════


class Expr:
    """A node of a program. .attr reaches into its value, calling it calls the value, .named gives it a label."""

    label: str

    def __getattr__(self, attr: str) -> Attr:
        if attr.startswith("__"):
            raise AttributeError(attr)
        return Attr(self, attr)

    def __call__(self, *args: Any) -> Call:
        return Call(self, args)

    def named(self, label: str) -> Any:
        return replace(self, label=label)  # type: ignore[type-var]


@dataclass(frozen=True, eq=False)
class Input(Expr):
    name: str
    label: str = ""


@dataclass(frozen=True, eq=False)
class Attr(Expr):
    base: Expr
    attr: str
    label: str = ""


@dataclass(frozen=True, eq=False)
class Call(Expr):
    fn: Any            # an entry (settings bound by functools.partial), a program, or a node holding a model
    args: tuple[Any, ...]
    label: str = ""

    @property
    def op(self) -> Any:
        """The function called, without its settings."""
        return self.fn.func if isinstance(self.fn, partial) else self.fn

    @property
    def settings(self) -> dict[str, Any]:
        return dict(self.fn.keywords) if isinstance(self.fn, partial) else {}

    def with_settings(self, **settings: Any) -> Call:
        bound = self.fn.args if isinstance(self.fn, partial) else ()
        return replace(self, fn=partial(self.op, *bound, **{**self.settings, **settings}))


def inputs(names: str) -> tuple[Input, ...]:
    return tuple(Input(n) for n in names.split())


def call(fn: Any, *args: Any, **settings: Any) -> Call:
    """fn applied to nodes (or constants), with settings bound."""
    return Call(partial(fn, **settings) if settings else fn, args)


def calls(op: Any) -> Callable[[Any], bool]:
    """Matches the calls of an entry (whatever its settings)."""
    return lambda e: isinstance(e, Call) and e.op is op


def _children(e: Any) -> list[Any]:
    if isinstance(e, Call):
        return [e.fn, *e.args]
    return [e.base] if isinstance(e, Attr) else []


def _reach(value: Any, attr: str) -> Any:
    if attr.lstrip("-").isdigit():
        return value[int(attr)]
    if hasattr(value, attr):
        return getattr(value, attr)
    return (value.data if isinstance(value, Container) else value)[attr]


@dataclass(frozen=True, eq=False)
class Program:
    """A named expression tree with inputs: calling it evaluates the tree, each shared node once."""

    name: str
    inputs: tuple[Input, ...]
    output: Expr
    about: str = ""

    def __call__(self, *args: Any) -> Any:
        if len(args) != len(self.inputs):
            raise TypeError(f"{self.name} takes ({', '.join(i.name for i in self.inputs)}), got {len(args)} values")
        memo: dict[int, Any] = {id(i): a for i, a in zip(self.inputs, args)}

        def run(e: Any) -> Any:
            if not isinstance(e, Expr):
                return e
            if id(e) not in memo:
                if isinstance(e, Input):
                    raise NameError(f"{self.name}: {e.name} is not one of its inputs")
                if isinstance(e, Attr):
                    memo[id(e)] = _reach(run(e.base), e.attr)
                elif isinstance(e, Call):
                    memo[id(e)] = run(e.fn)(*map(run, e.args))
            return memo[id(e)]

        return run(self.output)

    # ── modifying ──

    def nodes(self) -> list[Expr]:
        """Every node, children before parents, each once."""
        seen: dict[int, Expr] = {}

        def visit(e: Any) -> None:
            if isinstance(e, Expr) and id(e) not in seen:
                for c in _children(e):
                    visit(c)
                seen[id(e)] = e

        visit(self.output)
        return list(seen.values())

    def find(self, match: Callable[[Any], bool]) -> list[Expr]:
        return [e for e in self.nodes() if match(e)]

    def rewrite(self, match: Callable[[Any], bool], change: Callable[[Any], Any]) -> Program:
        """A new program with every matching node replaced by change(node), bottom-up; shared nodes stay shared, and
        programs used as constants (losses, rounds) are rewritten too."""
        memo: dict[int, Any] = {}

        def go(e: Any) -> Any:
            if isinstance(e, Program):
                return e.rewrite(match, change)
            if not isinstance(e, Expr):
                return e
            if id(e) not in memo:
                new = e
                if isinstance(e, Call):
                    fn, args = go(e.fn), tuple(go(a) for a in e.args)
                    if fn is not e.fn or any(a is not b for a, b in zip(args, e.args)):
                        new = replace(e, fn=fn, args=args)
                elif isinstance(e, Attr) and (base := go(e.base)) is not e.base:
                    new = replace(e, base=base)
                memo[id(e)] = change(new) if match(new) else new
            return memo[id(e)]

        return replace(self, output=go(self.output))

    # ── printing ──

    def __str__(self) -> str:
        printed: set[int] = set()
        rows: list[tuple[str, str]] = []

        def text(e: Any) -> str:
            if isinstance(e, Program):
                return e.name
            if not isinstance(e, Expr):
                return _short(e)
            if isinstance(e, Input):
                return e.name
            if isinstance(e, Attr):
                return e.label or f"{text(e.base)}.{e.attr}"
            return e.label or (text(e.fn) if isinstance(e.fn, Expr) else _name(e.fn)) + "(…)"

        def calls_below(e: Any) -> list[Call]:
            out = []
            for c in _children(e):
                out += [c] if isinstance(c, Call) else calls_below(c) if isinstance(c, Attr) else []
            return out

        def show(e: Call, depth: int) -> None:
            if id(e) in printed:
                return
            printed.add(id(e))
            name = text(e.fn) if isinstance(e.fn, Expr) else _name(e.fn)
            settings = _settings(e.fn)
            lhs = f"{e.label} = " if e.label else ""
            rows.append(("    " + "  " * depth + f"{lhs}{name}({', '.join(map(text, e.args))}{settings})", group_of(e.fn)))
            for c in calls_below(e):
                show(c, depth + 1)

        root = self.output
        while isinstance(root, Attr):
            root = root.base
        if isinstance(root, Call):
            show(root, 0)
        width = max(len(r[0]) for r in rows) if rows else 0
        lines = [f"{self.name}({', '.join(i.name for i in self.inputs)}) → {text(self.output)}"] + ([f"  {self.about}"] if self.about else [])
        lines += [f"{left:<{width}}   {g}" for left, g in rows]
        subs = {id(a): a for e in self.nodes() if isinstance(e, Call) for a in (e.fn, *e.args, *_bound(e.fn)) if isinstance(a, Program)}
        for sub in subs.values():
            lines += [f"  where {sub.name} is"] + ["    " + line for line in str(sub).splitlines()]
        return "\n".join(lines)


BUILDERS: set[Any] = set()  # functions that return a program (shown as "program")


def _bound(fn: Any) -> list[Any]:
    """What a partial binds (its arguments and settings); nothing for anything else."""
    return [*fn.args, *fn.keywords.values()] if isinstance(fn, partial) else []


def _name(op: Any) -> str:
    if isinstance(op, partial):
        return _name(op.func)
    return op.name if isinstance(op, Program) else getattr(op, "__name__", repr(op))


def _short(v: Any) -> str:
    return v.name if isinstance(v, Program) else getattr(v, "__name__", None) or repr(v)


def _settings(fn: Any) -> str:
    if not isinstance(fn, partial):
        return ""
    return "; " + ", ".join([*map(_short, fn.args), *(f"{k}={_short(v)}" for k, v in fn.keywords.items())])


def group_of(fn: Any) -> str:
    """The catalog group of a call: "model" for a node holding a model, "program" for a program, "·" for glue."""
    if isinstance(fn, Expr):
        return "model"
    op = fn.func if isinstance(fn, partial) else fn
    if isinstance(op, Program) or op in BUILDERS:
        return "program"
    e = entry_of(op)
    return e.group if e is not None else "·"


def groups(p: Program) -> list[str]:
    """The groups a program draws on, its sub-programs' included, in order of first use."""
    seen: list[str] = []
    for e in p.nodes():
        if isinstance(e, Call):
            g = group_of(e.fn)
            if g not in seen and g not in ("model", "program", "·"):
                seen.append(g)
            for a in (*e.args, *_bound(e.fn)):
                if isinstance(a, Program):
                    seen += [g for g in groups(a) if g not in seen]
    return seen


# ═══ Glue, not notions ════════════════════════════════════════════════════════════════════════════


def towers(image: nn.Module, text: nn.Module) -> nn.ModuleDict:
    """Two encoders, trained as one module."""
    return nn.ModuleDict({"image": image, "text": text})


def head(name: str) -> bool:
    """Whether a parameter belongs to the readout."""
    return name.startswith("readout")


# ═══ Losses, as programs: a split, a model, a score ═══════════════════════════════════════════════


def split_loss(name: str, split: Any, about: str) -> Program:
    """A loss of (model, data): split the data, predict from the input, score with the loss the target picks."""
    model, data = inputs("network data")
    ex = call(split, data).named("ex")
    return Program(name, (model, data), call(loss, model(ex.input), ex.target, ex.weight), about)


next_token = split_loss("next-token loss", shift, "each token predicted from the ones before it")
masked_loss = split_loss("masked loss", mask, "random holes, predicted")
autoencoding = split_loss("reconstruction loss", reconstruction, "the input as its own target, through the bottleneck")
dynamics_loss = split_loss("dynamics loss", dynamics, "from a state and an action, what follows")


def _supervised() -> Program:
    model, x, y = inputs("network x y")
    ex = call(labelled, x, y).named("ex")
    return Program("supervised loss", (model, x, y), call(loss, model(ex.input), ex.target), "the label as the target")


def _flow_matching() -> Program:
    velocity, z, labels = inputs("velocity z labels")
    ex = call(noise, z).named("ex")
    return Program("conditional flow-matching loss", (velocity, z, labels), call(loss, velocity(ex.input, labels), ex.target),
                   "noised latents; the velocity back, now and then without the label")


def _contrastive() -> Program:
    pair, images, texts = inputs("towers images texts")
    return Program("contrastive loss", (pair, images, texts), call(info_nce, pair.image(images), pair.text(texts)),
                   "which caption goes with which image, among the batch")


def _ppo_loss() -> Program:
    agent, batch = inputs("agent batch")
    logits, value = agent.policy(batch.observation).named("logits"), agent.critic(batch.observation).named("value")
    advantage = call(operator.sub, batch.returns, value).named("advantage")
    policy_loss = call(ppo_clip, call(log_prob, logits, batch.action), batch.log_prob, advantage).named("policy_loss")
    return Program("PPO loss", (agent, batch), call(operator.add, policy_loss, call(loss, value, batch.returns)),
                   "the clipped policy gradient with the critic as baseline, plus the critic's regression")


def _q_learning() -> Program:
    q, batch = inputs("q batch")
    ex = call(q_target, batch, q).named("ex")
    return Program("Q-learning loss", (q, batch), call(loss, q(ex.input), ex.target, ex.weight), "the action taken learns toward r + γ·max Q(s′)")


supervised, flow_matching, contrastive, ppo_loss, q_learning = _supervised(), _flow_matching(), _contrastive(), _ppo_loss(), _q_learning()


def grpo(verify: Callable[..., Any], n: int = 8, length: int = 1) -> Program:
    policy, prompts = inputs("policy prompts")
    path = call(walk, call(lm_tree, policy, prompts), call(independent, n, sample), steps=length).named("path")
    advantage = call(group_relative, call(verify, path.end), n=n)
    return Program("GRPO loss", (policy, prompts), call(reinforce, path, advantage),
                   "n answers per prompt, verified, each pushed by how it did against its group")


BUILDERS.add(grpo)

# ═══ Well-known work ══════════════════════════════════════════════════════════════════════════════


def language_model(name: str = "GPT", about: str = "a causal Transformer: shift, the loss, train; then walk the LM's tree",
                   build: Callable[..., Any] = network, steps: int = 200) -> Program:
    corpus, prompt = inputs("corpus prompt")
    ex = call(shift, corpus).named("ex")
    net = call(train, call(build, ex.input.known, ex.target), next_token, corpus, steps=steps).named("network")
    path = call(walk, call(lm_tree, net, prompt), choose=greedy, steps=8)
    return Program(name, (corpus, prompt), path.end, about)


def mamba() -> Program:
    return language_model("Mamba", "GPT with a selective scan in place of attention", partial(network, mixer=selective_scan))


def mixtral() -> Program:
    return language_model("Mixtral / DeepSeek-V3", "GPT whose MLP is a routed mixture of experts", partial(network, mlp=moe))


def vit(steps: int = 100) -> Program:
    images, labels = inputs("images labels")
    tokens = call(flatten, call(patchify, images, size=4)).named("tokens")
    net = call(train, call(network, tokens, labels), supervised, tokens, labels, steps=steps).named("network")
    return Program("ViT", (images, labels), call(accuracy, net(tokens), labels), "an image as a sequence of patches, read by a Transformer")


def clip(dim: int = 16, steps: int = 150) -> Program:
    images, texts = inputs("images texts")
    patches = call(patchify, images, size=4).named("patches")
    pair = call(train, call(towers, call(encoder, patches, dim=dim), call(encoder, texts, dim=dim)), contrastive, patches, texts,
                steps=steps).named("towers")
    return Program("CLIP", (images, texts), call(retrieval_accuracy, pair.image(patches), pair.text(texts)),
                   "an image encoder and a text encoder, trained so that each image picks out its caption")


def stable_diffusion(latent: int = 4, steps: int = 150, w: float = 3.0) -> Program:
    images, labels = inputs("images labels")
    ae = call(frozen_copy, call(train, call(autoencoder, images, latent=latent), autoencoding, images, steps=steps)).named("ae")
    z = ae.encode(images).named("z")
    ex = call(noise, z).named("ex")
    velocity = call(train, call(droppable, call(network, ex.input.known, ex.target), p=0.1), flow_matching, z, labels, steps=steps).named("velocity")
    guided_model = call(guided, call(given, velocity, labels), call(given, velocity, None), w=w)
    return Program("Stable Diffusion", (images, labels), ae.decode(call(fill, call(blank, z), guided_model, schedule=together, steps=10)),
                   "flow matching in an autoencoder's latent space, with a label, sampled with classifier-free guidance")


def lora_finetune(rank: int = 4, steps: int = 150) -> Program:
    net, corpus, prompt = inputs("network corpus prompt")
    tuned = call(train, call(lora, net, rank=rank), next_token, corpus, steps=steps).named("network")
    return Program("LoRA finetuning", (net, corpus, prompt), call(walk, call(lm_tree, tuned, prompt), choose=greedy, steps=8).end,
                   "a pretrained model, frozen, with a trainable low-rank update")


def dpo_program(beta: float = 0.1, steps: int = 60) -> Program:
    policy, chosen, rejected = inputs("policy chosen rejected")
    objective = call(dpo, call(frozen_copy, policy), beta=beta)
    return Program("DPO", (policy, chosen, rejected), call(train, policy, objective, chosen, rejected, steps=steps),
                   "preference pairs as a classification of log-ratios against a frozen reference")


def alphazero(rounds: int = 8, simulations: int = 32, steps: int = 30, length: int = 16) -> Program:
    net, game = inputs("network game")
    search = call(mcts, game(net), simulations=simulations, two_player=True).named("search")
    ex = call(search_targets, call(walk, search, choose=sample, steps=length), search).named("ex")
    round_ = Program("self-play round", (net, game), call(train, net, supervised, ex.input.known, ex.target, steps=steps),
                     "play yourself with search; learn the search's choices and each game's outcome")
    net2, game2 = inputs("network game")
    trained = call(loop, round_, net2, game2, times=rounds).named("network")
    return Program("AlphaZero", (net2, game2), call(walk, game2(trained), choose=greedy, steps=length),
                   "a policy-value network that amortizes MCTS, trained by self-play")


def r1(n: int = 8, length: int = 1, steps: int = 100) -> Program:
    policy, prompts, verify = inputs("policy prompts verify")
    trained = call(train, policy, call(grpo, verify, n=n, length=length), prompts, steps=steps).named("policy")
    path = call(walk, call(lm_tree, trained, prompts), choose=greedy, steps=length)
    return Program("DeepSeek-R1 (GRPO)", (policy, prompts, verify), verify(path.end),
                   "RL on checkable answers: walk, verify, group-relative advantages, REINFORCE")


def diffusion_lm(steps: int = 600) -> Program:
    corpus, = inputs("corpus")
    ex = call(mask, corpus).named("ex")
    net = call(train, call(network, ex.input.known, ex.target), masked_loss, corpus, steps=steps).named("network")
    return Program("Diffusion LLM", (corpus,), call(fill, call(blank, corpus), net, schedule=most_confident, steps=8),
                   "masked prediction at every ratio; generation unmasks the surest holes first")


# ═══ Compositions the catalog points to ═══════════════════════════════════════════════════════════


def pretrain_finetune(steps: int = 150) -> Program:
    corpus, texts, labels = inputs("corpus texts labels")
    ex = call(shift, corpus).named("ex")
    lm = call(train, call(network, ex.input.known, ex.target), next_token, corpus, steps=steps).named("lm")
    body = call(freeze, call(transfer, lm, call(network, texts, labels)), trainable=head).named("classifier")
    return Program("Pretrain → finetune", (corpus, texts, labels), call(train, body, supervised, texts, labels, steps=steps),
                   "learn from raw text, then reuse the body under a new head")


def _rl(name: str, about: str, round_: Program, rounds: int, horizon: int) -> Program:
    agent, env = inputs("agent env")
    trained = call(loop, round_, agent, env, times=rounds).named("agent")
    policy = trained.policy if name.startswith("PPO") else trained
    return Program(name, (agent, env), call(discounted_return, call(walk, call(env_tree, env, policy), choose=greedy, steps=horizon)), about)


def ppo(rounds: int = 30, horizon: int = 20, epochs: int = 4) -> Program:
    agent, env = inputs("agent env")
    path = call(walk, call(env_tree, env, agent.policy), choose=sample, steps=horizon).named("path")
    batch = call(transitions, path, call(returns_to_go, path.rewards, gamma=0.9)).named("batch")
    round_ = Program("PPO round", (agent, env), call(train, agent, ppo_loss, batch, steps=epochs),
                     "walk the environment, then a few clipped steps on what was seen")
    return _rl("PPO (actor-critic)", "a policy and a critic, improved round by round on their own walks", round_, rounds, horizon)


def dqn(rounds: int = 30, horizon: int = 20, steps: int = 10, eps: float = 0.3) -> Program:
    q, env = inputs("q env")
    path = call(walk, call(epsilon_greedy, call(env_tree, env, q), eps=eps), choose=sample, steps=horizon).named("path")
    round_ = Program("DQN round", (q, env), call(train, q, q_learning, call(transitions, path), steps=steps),
                     "walk ε-greedily on the Q-values, then regress them toward their targets")
    return _rl("DQN", "values learned by bootstrapping, acted on greedily", round_, rounds, horizon)


def world_model_planning(horizon: int = 20, steps: int = 300) -> Program:
    env, = inputs("env")
    tree = call(env_tree, env).named("tree")
    batch = call(transitions, call(walk, tree, choose=sample, steps=horizon)).named("batch")
    ex = call(dynamics, batch).named("ex")
    world = call(train, call(network, ex.input.known, ex.target), dynamics_loss, batch, steps=steps).named("world")
    plan = call(planned, tree, call(imagined, tree, world), horizon=8, samples=32).named("plan")
    return Program("Planning in a learned world model", (env,), call(discounted_return, call(walk, plan, choose=greedy, steps=horizon)),
                   "explore at random, learn what follows each action, act by planning inside that model (Dreamer, MuZero)")


def synthetic_data(n: int = 8, length: int = 4, keep: float = 0.25) -> Program:
    generator, prompts, verify = inputs("generator prompts verify")
    path = call(walk, call(temperature, call(lm_tree, generator, prompts), t=1.5), call(independent, n, sample), steps=length).named("path")
    return Program("Synthetic data", (generator, prompts, verify), call(quality_filter, call(Seq, path.end), verify(path.end), keep=keep),
                   "sample a model's tree widely, keep what a verifier scores best")


def rag(k: int = 4) -> Program:
    lm, embed, documents, question = inputs("lm embed documents question")
    memory = call(index, embed(documents), kind="hnsw").named("memory")
    prompt = call(retrieve, memory, documents, question, embed, k=k).named("prompt")
    return Program("Retrieval-augmented generation", (lm, embed, documents, question),
                   call(walk, call(lm_tree, lm, prompt), choose=greedy, steps=64).end,
                   "index the documents' embeddings, put the nearest ones in front of the question, decode")


PROGRAMS: list[Program] = [
    language_model(), vit(), clip(), stable_diffusion(), mamba(), mixtral(), lora_finetune(), dpo_program(), alphazero(), r1(), diffusion_lm(),
    pretrain_finetune(), ppo(), dqn(), world_model_planning(), synthetic_data(), rag(),
]

if __name__ == "__main__":
    for p in PROGRAMS:
        print(p, end="\n\n")
