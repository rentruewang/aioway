"""
program.py: functional and imperative, split as a database splits them.

A Program is a query: a pure expression tree over the entries of the catalog. It builds values (data, new models,
losses) and changes nothing.

    corpus, prompt = inputs("corpus prompt")
    ex = call(shift, corpus).named("ex")                       # a call: an entry applied to other nodes
    net = call(network, ex.input.known, ex.target)             # .attr reaches into a node's value
    out = net(prompt)                                          # calling a node calls the value it stands for

A Script is the imperative part: an ordered list of statements over named values.

    let(net, expr)                      net = expr                    pure: a name for a value
    update(net, train, loss, data)      train(net!, loss, data)       changes net in place, returns None
    repeat(n, *statements)              the statements, n times

An update is one of the catalog's statements (its imperative part: fit, assign, alter). It changes its model (!)
in place and returns None, so it is a line of its own, never a value inside an expression, as a write in SQL sits at the top of a statement and never inside
a query. The name still names the same model, now changed. A statement evaluates its arguments first, then runs:
data read from the model before is a snapshot for free; a model built on it (lm_tree, given, a reference) is the
same model and sees the change, so one that must stay as it was (DPO's reference) is a frozen_copy, let first.

Both are trees, and modifying one is rewriting it, which returns a new one (the old one is unchanged):

    GPT.rewrite(calls(walk), lambda n: n.with_settings(steps=32))            # change one node's settings
    GPT.rewrite(calls(network), lambda n: call(network, *n.args, mixer=selective_scan))   # swap a subtree
    GPT.find(calls(train))                                                   # find nodes

check() lists what breaks the split: an update inside an expression, a name read before it is bound.

Printing shows each statement's tree from its root down, each call with its catalog group. The entries are
interfaces: programs and scripts print and rewrite now, and run once the entries are implemented.

    python program.py      prints every program and script
"""

from __future__ import annotations

import operator
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field, replace
from functools import partial
from typing import Any, Self

from torch import nn

from catalog import (
    accuracy, autoencoder, blank, changes, discounted_return, dpo, droppable, dynamics, encoder, entry_of, env_tree, epsilon_greedy, fill, flatten,
    freeze, frozen_copy, given, greedy, group_relative, guided, imagined, independent, index, info_nce, labelled, lm_tree, log_prob, lora,
    loss, mask, mcts, moe, most_confident, network, noise, patchify, planned, ppo_clip, q_target, quality_filter, reconstruction, reinforce,
    retrieval_accuracy, retrieve, returns_to_go, sample, search_targets, selective_scan, shift, temperature, together, train, transfer,
    transitions, walk,
)
from data import Container, Seq

# ═══ Expressions ══════════════════════════════════════════════════════════════════════════════════


@dataclass(frozen=True, eq=False)
class Expr:
    """A node of an expression. .attr reaches into its value, calling it calls the value, .named gives it a label."""

    label: str = field(default="", kw_only=True)

    def __getattr__(self, attr: str) -> Attr:
        if attr.startswith("__"):
            raise AttributeError(attr)
        return Attr(self, attr)

    def __call__(self, *args: Any) -> Call:
        return Call(self, args)

    def named(self, label: str) -> Self:
        return replace(self, label=label)


@dataclass(frozen=True, eq=False)
class Input(Expr):
    """A name: a program's input, or a script's variable."""

    name: str


@dataclass(frozen=True, eq=False)
class Attr(Expr):
    base: Expr
    attr: str


@dataclass(frozen=True, eq=False)
class Call(Expr):
    fn: Any            # an entry (settings bound by functools.partial), a program, or a node holding a model
    args: tuple[Any, ...]

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


def _evaluate(root: Any, env: dict[str, Any], where: str) -> Any:
    """The value of an expression, its names read from env, each shared node once."""
    memo: dict[int, Any] = {}

    def run(e: Any) -> Any:
        if not isinstance(e, Expr):
            return e
        if id(e) not in memo:
            if isinstance(e, Input):
                if e.name not in env:
                    raise NameError(f"{where}: {e.name} is not bound")
                memo[id(e)] = env[e.name]
            elif isinstance(e, Attr):
                memo[id(e)] = _reach(run(e.base), e.attr)
            elif isinstance(e, Call):
                memo[id(e)] = run(e.fn)(*map(run, e.args))
        return memo[id(e)]

    return run(root)


def _nodes(roots: list[Any]) -> list[Expr]:
    """Every node below the roots, children before parents, each once."""
    seen: dict[int, Expr] = {}

    def visit(e: Any) -> None:
        if isinstance(e, Expr) and id(e) not in seen:
            for c in _children(e):
                visit(c)
            seen[id(e)] = e

    for r in roots:
        visit(r)
    return list(seen.values())


def _rewriter(match: Callable[[Any], bool], change: Callable[[Any], Any]) -> Callable[[Any], Any]:
    """Rewrites expressions bottom-up, every matching node replaced by change(node); shared nodes stay shared (one
    memo for every expression it is given), and programs used as constants (losses) are rewritten too."""
    memo: dict[int, Any] = {}

    def go(e: Any) -> Any:
        if isinstance(e, (Program, Script)):
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

    return go


def _updates_inside(where: str, roots: list[Any], allowed: Any = None) -> list[str]:
    """An update called inside an expression (other than the statement's own): the split, broken."""
    return [f"{where}: {_name(e.fn)} changes a model inside an expression; make it a statement (update)"
            for e in _nodes(roots) if isinstance(e, Call) and e is not allowed and changes(e.op) is not None]


def _sub_checks(roots: list[Any]) -> list[str]:
    subs = {id(a): a for e in _nodes(roots) if isinstance(e, Call) for a in (e.fn, *e.args, *_bound(e.fn)) if isinstance(a, (Program, Script))}
    return [c for sub in subs.values() for c in sub.check()]


# ═══ Program: a query ═════════════════════════════════════════════════════════════════════════════


@dataclass(frozen=True, eq=False)
class Program:
    """A named pure expression with inputs: calling it evaluates the tree, each shared node once."""

    name: str
    inputs: tuple[Input, ...]
    output: Expr
    about: str = ""

    def __call__(self, *args: Any) -> Any:
        if len(args) != len(self.inputs):
            raise TypeError(f"{self.name} takes ({', '.join(i.name for i in self.inputs)}), got {len(args)} values")
        return _evaluate(self.output, {i.name: a for i, a in zip(self.inputs, args)}, self.name)

    def nodes(self) -> list[Expr]:
        return _nodes([self.output])

    def find(self, match: Callable[[Any], bool]) -> list[Expr]:
        return [e for e in self.nodes() if match(e)]

    def rewrite(self, match: Callable[[Any], bool], change: Callable[[Any], Any]) -> Program:
        return replace(self, output=_rewriter(match, change)(self.output))

    def check(self) -> list[str]:
        """A query changes nothing: no update anywhere in it, nor in the programs it uses."""
        return _updates_inside(self.name, [self.output]) + _sub_checks([self.output])

    def __str__(self) -> str:
        lines = [f"{self.name}({', '.join(i.name for i in self.inputs)}) → {_text(self.output)}"] + ([f"  {self.about}"] if self.about else [])
        return "\n".join(lines + _columns(_tree(self.output, "")) + _where([self.output]))


# ═══ Script: the imperative part ══════════════════════════════════════════════════════════════════


@dataclass(frozen=True)
class Let:
    """name = value: a name for the value of a pure expression."""

    name: Input
    value: Any


@dataclass(frozen=True)
class Update:
    """op(…, name!, …): an update entry, changing the named model in place (it returns None)."""

    name: Input
    call: Call


@dataclass(frozen=True)
class Repeat:
    """The statements, in order, times times."""

    times: int
    body: tuple[Statement, ...]


type Statement = Let | Update | Repeat


def let(name: Input, value: Any) -> Let:
    return Let(name, value)


def update(name: Input, op: Any, *args: Any, **settings: Any) -> Update:
    """op(args) with name put where op takes the model it changes (train: first; transfer: the target)."""
    i = changes(op)
    if i is None:
        raise TypeError(f"{_name(op)} is functional; an update needs a statement of the catalog (fit, assign, alter)")
    return Update(name, call(op, *args[:i], name, *args[i:], **settings))


def repeat(times: int, *body: Statement) -> Repeat:
    return Repeat(times, body)


def _flat(body: tuple[Statement, ...]) -> Iterator[Let | Update]:
    for s in body:
        yield from _flat(s.body) if isinstance(s, Repeat) else (s,)


def _expr(s: Let | Update) -> Any:
    return s.value if isinstance(s, Let) else s.call


@dataclass(frozen=True, eq=False)
class Script:
    """Statements over named values, in order, then a pure expression of the names as they end up."""

    name: str
    inputs: tuple[Input, ...]
    body: tuple[Statement, ...]
    output: Any
    about: str = ""

    def __call__(self, *args: Any) -> Any:
        if len(args) != len(self.inputs):
            raise TypeError(f"{self.name} takes ({', '.join(i.name for i in self.inputs)}), got {len(args)} values")
        env = {i.name: a for i, a in zip(self.inputs, args)}

        def run(body: tuple[Statement, ...]) -> None:
            for s in body:
                if isinstance(s, Repeat):
                    for _ in range(s.times):
                        run(s.body)
                elif isinstance(s, Let):
                    env[s.name.name] = _evaluate(s.value, env, f"{self.name}: {s.name.name}")
                else:  # the arguments first, then the update, in place: the name still names the same model
                    _evaluate(s.call, env, f"{self.name}: {s.name.name}")

        run(self.body)
        return _evaluate(self.output, env, self.name)

    def _roots(self) -> list[Any]:
        return [*map(_expr, _flat(self.body)), self.output]

    def nodes(self) -> list[Expr]:
        return _nodes(self._roots())

    def find(self, match: Callable[[Any], bool]) -> list[Expr]:
        return [e for e in self.nodes() if match(e)]

    def rewrite(self, match: Callable[[Any], bool], change: Callable[[Any], Any]) -> Script:
        go = _rewriter(match, change)

        def statements(body: tuple[Statement, ...]) -> tuple[Statement, ...]:
            return tuple(Repeat(s.times, statements(s.body)) if isinstance(s, Repeat) else
                         Let(s.name, go(s.value)) if isinstance(s, Let) else Update(s.name, go(s.call)) for s in body)

        return replace(self, body=statements(self.body), output=go(self.output))

    def check(self) -> list[str]:
        """Updates only as statements, each of an update entry on its own name; every name bound before it is read."""
        out: list[str] = []
        bound = {i.name for i in self.inputs}
        for s in _flat(self.body):
            where = f"{self.name}: {s.name.name}"
            if isinstance(s, Update):
                i = changes(s.call.op)
                if i is None or i >= len(s.call.args) or s.call.args[i] is not s.name:
                    out.append(f"{where}: {_name(s.call.fn)} is not an update of {s.name.name}")
            out += _updates_inside(where, [_expr(s)], s.call if isinstance(s, Update) else None)
            out += [f"{where}: reads {e.name} before it is bound" for e in _nodes([_expr(s)]) if isinstance(e, Input) and e.name not in bound]
            bound.add(s.name.name)
        out += _updates_inside(self.name, [self.output])
        out += [f"{self.name}: reads {e.name}, never bound" for e in _nodes([self.output]) if isinstance(e, Input) and e.name not in bound]
        return out + _sub_checks(self._roots())

    def __str__(self) -> str:
        rows: list[tuple[str, str]] = []

        def show(body: tuple[Statement, ...], depth: int) -> None:
            pad = "  " * depth
            for s in body:
                if isinstance(s, Repeat):
                    rows.append((f"    {pad}repeat ×{s.times}:", ""))
                    show(s.body, depth + 1)
                elif isinstance(s, Let):
                    rows.extend(_tree(s.value, pad, f"{s.name.name} = "))
                else:
                    rows.extend(_tree(s.call, pad, changes=s.name))

        show(self.body, 0)
        rows.extend(_tree(self.output, "", "return "))
        lines = [f"{self.name}({', '.join(i.name for i in self.inputs)}) → {_text(self.output)}"] + ([f"  {self.about}"] if self.about else [])
        return "\n".join(lines + _columns(rows) + _where(self._roots()))


# ═══ Printing ═════════════════════════════════════════════════════════════════════════════════════

BUILDERS: set[Any] = set()  # functions that return a program (shown as "program")


def _bound(fn: Any) -> list[Any]:
    """What a partial binds (its arguments and settings); nothing for anything else."""
    return [*fn.args, *fn.keywords.values()] if isinstance(fn, partial) else []


def _name(op: Any) -> str:
    if isinstance(op, partial):
        return _name(op.func)
    return op.name if isinstance(op, (Program, Script)) else getattr(op, "__name__", repr(op))


def _short(v: Any) -> str:
    return v.name if isinstance(v, (Program, Script)) else getattr(v, "__name__", None) or repr(v)


def _settings(fn: Any) -> str:
    if not isinstance(fn, partial):
        return ""
    return "; " + ", ".join([*map(_short, fn.args), *(f"{k}={_short(v)}" for k, v in fn.keywords.items())])


def _text(e: Any) -> str:
    """A node as an argument: its label, or a short form."""
    if not isinstance(e, Expr):
        return _short(e)
    if isinstance(e, Input):
        return e.name
    if isinstance(e, Attr):
        return e.label or f"{_text(e.base)}.{e.attr}"
    assert isinstance(e, Call)
    return e.label or (_text(e.fn) if isinstance(e.fn, Expr) else _name(e.fn)) + "(…)"


def _tree(root: Any, pad: str, lhs: str = "", changes: Input | None = None) -> list[tuple[str, str]]:
    """Rows (text, group) for the calls of an expression, from its root down, each shared call once. changes is the
    name an update changes, marked ! in its root call."""
    printed: set[int] = set()
    rows: list[tuple[str, str]] = []

    def calls_below(e: Any) -> list[Call]:
        out = []
        for c in _children(e):
            out += [c] if isinstance(c, Call) else calls_below(c) if isinstance(c, Attr) else []
        return out

    def show(e: Call, depth: int, lhs: str) -> None:
        if id(e) in printed:
            return
        printed.add(id(e))
        name = _text(e.fn) if isinstance(e.fn, Expr) else _name(e.fn)
        args = ", ".join(_text(a) + ("!" if depth == 0 and a is changes else "") for a in e.args)
        rows.append((f"    {pad}{'  ' * depth}{lhs}{name}({args}{_settings(e.fn)})", group_of(e.fn)))
        for c in calls_below(e):
            show(c, depth + 1, f"{c.label} = " if c.label else "")

    top, path = root, ""
    while isinstance(top, Attr):
        top, path = top.base, f".{top.attr}{path}"
    if isinstance(top, Call):
        show(top, 0, lhs if not top.label or lhs else f"{top.label} = ")
        if path and lhs:
            rows[0] = (rows[0][0] + path, rows[0][1])
    elif lhs:
        rows.append((f"    {pad}{lhs}{_text(root)}", ""))
    return rows


def _columns(rows: list[tuple[str, str]]) -> list[str]:
    width = max((len(r[0]) for r in rows), default=0)
    return [f"{left:<{width}}   {g}".rstrip() for left, g in rows]


def _where(roots: list[Any]) -> list[str]:
    subs = {id(a): a for e in _nodes(roots) if isinstance(e, Call) for a in (e.fn, *e.args, *_bound(e.fn)) if isinstance(a, (Program, Script))}
    return [line for sub in subs.values() for line in [f"  where {sub.name} is", *("    " + l for l in str(sub).splitlines())]]


def group_of(fn: Any) -> str:
    """The catalog group of a call: "model" for a node holding a model, "program" for a program, "·" for glue."""
    if isinstance(fn, Expr):
        return "model"
    op = fn.func if isinstance(fn, partial) else fn
    if isinstance(op, (Program, Script)) or op in BUILDERS:
        return "program"
    e = entry_of(op)
    return e.group if e is not None else "·"


def groups(p: Program | Script) -> list[str]:
    """The groups a program or script draws on, its sub-programs' included, in order of first use."""
    seen: list[str] = []
    for e in p.nodes():
        if isinstance(e, Call):
            g = group_of(e.fn)
            if g not in seen and g not in ("model", "program", "·"):
                seen.append(g)
            for a in (*e.args, *_bound(e.fn)):
                if isinstance(a, (Program, Script)):
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
# Training is an update, so anything that trains is a script; the losses above, and what only reads a model, are
# programs.


def language_model(name: str = "GPT", about: str = "a causal Transformer: shift, the loss, train; then walk the LM's tree",
                   build: Callable[..., Any] = network, steps: int = 200) -> Script:
    corpus, prompt, net = inputs("corpus prompt network")
    ex = call(shift, corpus).named("ex")
    return Script(name, (corpus, prompt), (
        let(net, call(build, ex.input.known, ex.target)),
        update(net, train, next_token, corpus, steps=steps),
    ), call(walk, call(lm_tree, net, prompt), choose=greedy, steps=8).end, about)


def mamba() -> Script:
    return language_model("Mamba", "GPT with a selective scan in place of attention", partial(network, mixer=selective_scan))


def mixtral() -> Script:
    return language_model("Mixtral / DeepSeek-V3", "GPT whose MLP is a routed mixture of experts", partial(network, mlp=moe))


def vit(steps: int = 100) -> Script:
    images, labels, tokens, net = inputs("images labels tokens network")
    return Script("ViT", (images, labels), (
        let(tokens, call(flatten, call(patchify, images, size=4))),
        let(net, call(network, tokens, labels)),
        update(net, train, supervised, tokens, labels, steps=steps),
    ), call(accuracy, net(tokens), labels), "an image as a sequence of patches, read by a Transformer")


def clip(dim: int = 16, steps: int = 150) -> Script:
    images, texts, patches, pair = inputs("images texts patches towers")
    return Script("CLIP", (images, texts), (
        let(patches, call(patchify, images, size=4)),
        let(pair, call(towers, call(encoder, patches, dim=dim), call(encoder, texts, dim=dim))),
        update(pair, train, contrastive, patches, texts, steps=steps),
    ), call(retrieval_accuracy, pair.image(patches), pair.text(texts)),
        "an image encoder and a text encoder, trained so that each image picks out its caption")


def stable_diffusion(latent: int = 4, steps: int = 150, w: float = 3.0) -> Script:
    images, labels, ae, z, velocity = inputs("images labels ae z velocity")
    ex = call(noise, z).named("ex")
    sampler = call(guided, call(given, velocity, labels), call(given, velocity, None), w=w)
    return Script("Stable Diffusion", (images, labels), (
        let(ae, call(autoencoder, images, latent=latent)),
        update(ae, train, autoencoding, images, steps=steps),
        let(z, ae.encode(images)),
        let(velocity, call(droppable, call(network, ex.input.known, ex.target), p=0.1)),
        update(velocity, train, flow_matching, z, labels, steps=steps),
    ), ae.decode(call(fill, call(blank, z), sampler, schedule=together, steps=10)),
        "flow matching in an autoencoder's latent space, with a label, sampled with classifier-free guidance")


def lora_finetune(rank: int = 4, steps: int = 150) -> Script:
    net, corpus, prompt = inputs("network corpus prompt")
    return Script("LoRA finetuning", (net, corpus, prompt), (
        update(net, lora, rank=rank),
        update(net, train, next_token, corpus, steps=steps),
    ), call(walk, call(lm_tree, net, prompt), choose=greedy, steps=8).end, "a pretrained model, frozen, with a trainable low-rank update")


def dpo_script(beta: float = 0.1, steps: int = 60) -> Script:
    policy, chosen, rejected, reference = inputs("policy chosen rejected reference")
    return Script("DPO", (policy, chosen, rejected), (
        let(reference, call(frozen_copy, policy)),
        update(policy, train, call(dpo, reference, beta=beta), chosen, rejected, steps=steps),
    ), policy, "preference pairs as a classification of log-ratios against a frozen reference")


def alphazero(rounds: int = 8, simulations: int = 32, steps: int = 30, length: int = 16) -> Script:
    net, game, ex = inputs("network game ex")
    search = call(mcts, game(net), simulations=simulations, two_player=True).named("search")
    return Script("AlphaZero", (net, game), (
        repeat(rounds,
               let(ex, call(search_targets, call(walk, search, choose=sample, steps=length), search)),
               update(net, train, supervised, ex.input.known, ex.target, steps=steps)),
    ), call(walk, game(net), choose=greedy, steps=length),
        "a policy-value network that amortizes MCTS: each round plays itself with search, then learns the search's choices")


def r1(n: int = 8, length: int = 1, steps: int = 100) -> Script:
    policy, prompts, verify = inputs("policy prompts verify")
    return Script("DeepSeek-R1 (GRPO)", (policy, prompts, verify), (
        update(policy, train, call(grpo, verify, n=n, length=length), prompts, steps=steps),
    ), verify(call(walk, call(lm_tree, policy, prompts), choose=greedy, steps=length).end),
        "RL on checkable answers: walk, verify, group-relative advantages, REINFORCE")


def diffusion_lm(steps: int = 600) -> Script:
    corpus, net = inputs("corpus network")
    ex = call(mask, corpus).named("ex")
    return Script("Diffusion LLM", (corpus,), (
        let(net, call(network, ex.input.known, ex.target)),
        update(net, train, masked_loss, corpus, steps=steps),
    ), call(fill, call(blank, corpus), net, schedule=most_confident, steps=8),
        "masked prediction at every ratio; generation unmasks the surest holes first")


# ═══ Compositions the catalog points to ═══════════════════════════════════════════════════════════


def pretrain_finetune(steps: int = 150) -> Script:
    corpus, texts, labels, lm, clf = inputs("corpus texts labels lm classifier")
    ex = call(shift, corpus).named("ex")
    return Script("Pretrain → finetune", (corpus, texts, labels), (
        let(lm, call(network, ex.input.known, ex.target)),
        update(lm, train, next_token, corpus, steps=steps),
        let(clf, call(network, texts, labels)),
        update(clf, transfer, lm),
        update(clf, freeze, trainable=head),
        update(clf, train, supervised, texts, labels, steps=steps),
    ), clf, "learn from raw text, then reuse the body under a new head")


def ppo(rounds: int = 30, horizon: int = 20, epochs: int = 4) -> Script:
    agent, env, batch = inputs("agent env batch")
    path = call(walk, call(env_tree, env, agent.policy), choose=sample, steps=horizon).named("path")
    return Script("PPO (actor-critic)", (agent, env), (
        repeat(rounds,
               let(batch, call(transitions, path, call(returns_to_go, path.rewards, gamma=0.9))),
               update(agent, train, ppo_loss, batch, steps=epochs)),
    ), call(discounted_return, call(walk, call(env_tree, env, agent.policy), choose=greedy, steps=horizon)),
        "a policy and a critic; each round walks the environment, then takes a few clipped steps on what was seen")


def dqn(rounds: int = 30, horizon: int = 20, steps: int = 10, eps: float = 0.3) -> Script:
    q, env, batch = inputs("q env batch")
    return Script("DQN", (q, env), (
        repeat(rounds,
               let(batch, call(transitions, call(walk, call(epsilon_greedy, call(env_tree, env, q), eps=eps), choose=sample, steps=horizon))),
               update(q, train, q_learning, batch, steps=steps)),
    ), call(discounted_return, call(walk, call(env_tree, env, q), choose=greedy, steps=horizon)),
        "values learned by bootstrapping: each round walks ε-greedily, then regresses the Q-values toward their targets")


def world_model_planning(horizon: int = 20, steps: int = 300) -> Script:
    env, batch, world = inputs("env batch world")
    tree = call(env_tree, env).named("tree")
    ex = call(dynamics, batch).named("ex")
    plan = call(planned, tree, call(imagined, tree, world), horizon=8, samples=32).named("plan")
    return Script("Planning in a learned world model", (env,), (
        let(batch, call(transitions, call(walk, tree, choose=sample, steps=horizon))),
        let(world, call(network, ex.input.known, ex.target)),
        update(world, train, dynamics_loss, batch, steps=steps),
    ), call(discounted_return, call(walk, plan, choose=greedy, steps=horizon)),
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


PROGRAMS: list[Program | Script] = [
    language_model(), vit(), clip(), stable_diffusion(), mamba(), mixtral(), lora_finetune(), dpo_script(), alphazero(), r1(), diffusion_lm(),
    pretrain_finetune(), ppo(), dqn(), world_model_planning(), synthetic_data(), rag(),
]

if __name__ == "__main__":
    for p in PROGRAMS:
        print(p, end="\n\n")
