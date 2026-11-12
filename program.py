"""
program.py: expressions and statements, split as a database splits queries and writes.

A Program is an expression: a pure tree over the catalog's expressions, on batches in memory. It builds values
(data, new models, losses) and changes nothing. A node calls a config (a method with its settings) on other nodes:

    corpus, prompt = inputs("corpus prompt")
    ex = call(NextToken(), corpus).named("ex")                     # a config, called on a node
    net = call(Network, target=ex.target, d=64)                # .attr reaches into a node's value
    out = net(prompt)                                          # calling a node calls the value it stands for

A Script is statements in order, over named values, then an expression of what they end up as. A model is a layer
(catalog.LayerOp): making one allocates its parameters, so it is bound to a name, which statements change and later
expressions call: net = Network(target=y), then Train(…)(net!, …), then net(prompt).

    let(net, expr)                          net = expr                       a name for a value (or a dataset)
    update(net, Train(steps=200), loss, ds) Train(steps=200)(net!, loss, ds) changes net in place, returns None
    repeat(n, *statements)                  for _ in range(n): …             the statements, n times
    repeat(n, *statements, until=stop)      … if stop: break                 …at most n times, until stop is true

The catalog's commands (TrainCmd, DeployCmd, IndexCmd, …: they change their argument marked !; CodecCmd, OpCmd,
PartitionCmd: they make state or read a whole dataset) are lines of their own, never inside an expression, as a SQL write or CREATE INDEX sits
at the top of a statement. A statement's arguments are expressions (on batches) or dataset statements, nested as
a FROM nests: Train(net!, loss, Batches(x, y; size=64)). An update returns None and the name still names the same
model, now changed. A statement evaluates its arguments first, then runs: data read from the model before is a
snapshot for free; a model built on it (LMTree, Given, a reference) is the same model and sees the change, so one
that must stay as it was (DPO's reference) is a FrozenCopy, let first.

A program may leave a choice open with an intent (catalog.Intent): a config that stands for some method of a group,
AugmentImage() where RandomFlip(p=0.5) could be. It prints with a ?, and resolve.py lowers it to a concrete method.

A setting that is only known when the program runs (DPO's reference model) makes its config in the tree: calling a
class makes one, call(DPO, reference=reference, beta=0.1).

Both are trees, and modifying one is rewriting it, which returns a new one (the old one is unchanged):

    GPT.rewrite(calls(Walk), lambda n: n.with_settings(steps=32))               # change one node's settings
    GPT.rewrite(calls(Network), lambda n: n.with_settings(mixer=SSM(d=64)))   # swap a layer
    GPT.find(calls(Train))                                                      # find nodes

check() lists what breaks the split: a statement inside an expression, a change outside an update, a name read
before it is bound.

Printing shows each statement's tree from its root down, each call with its catalog group. The methods are
interfaces: programs and scripts print and rewrite now, and run once the methods are implemented.

    python program.py      prints every program and script
"""

from __future__ import annotations

import operator
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field, fields, is_dataclass, replace
from typing import Any, Self

from torch import nn

from catalog import (
    Accuracy, Autoencoder, Batches, Blank, Condition, Curriculum, Dedup, DiscountedReturn, DPO, CFGDropout, NearDedup,
    NewIndex, Encoder, EnvTree, EpsilonGreedy, Evaluate, Extend, Fill, Flatten, Freeze, FrozenCopy, Given, Greedy, GroupAdvantage, CFG, TrainTestSplit,
    WorldModel, NPaths, InfoNCE, Insert, KeepBest, LMTree, LogProb, LoRA, Loss, Mask, MCTS, Method, Mix, MoE, MostConfident,
    LayerOp, Network, Noise, Patchify, MPC, PPO, QualityFilter, Reconstruction, Reinforce, ReplayBuffer, RetrievalAccuracy,
    Retrieve, ReturnsToGo, Sample, SSM, NextToken, Temperature, AllAtOnce, Train, TrainIndex, Transfer,
    Transitions, Walk, changes, group_class, Intent, is_statement, settings_of, short,
)
from data import Container, Example, PartialTree, Path, Predictor, Seq, Temporal

# ═══ Nodes ════════════════════════════════════════════════════════════════════════════════════════


@dataclass(frozen=True, eq=False)
class Node:
    """A node of a tree. .attr reaches into its value, calling it calls the value, .named gives it a label."""

    label: str = field(default="", kw_only=True)

    def __getattr__(self, attr: str) -> Attr:
        if attr.startswith("__"):
            raise AttributeError(attr)
        return Attr(self, attr)

    def __call__(self, *args: Any, **kwargs: Any) -> Call:
        return Call(self, args, kwargs)

    def __getitem__(self, i: int) -> Attr:
        return Attr(self, str(i))

    def named(self, label: str) -> Self:
        return replace(self, label=label)


@dataclass(frozen=True, eq=False)
class Input(Node):
    """A name: a program's input, or a script's variable."""

    name: str


@dataclass(frozen=True, eq=False)
class Attr(Node):
    """base.attr: a field of a node's value (ex.target), or an element of it (splits[0], its attr "0")."""

    base: Node
    attr: str


@dataclass(frozen=True, eq=False)
class Call(Node):
    """fn(*args, **kwargs), its arguments nodes or constants. fn is a config (a method with its settings: it runs),
    a class (calling it makes one: a config, a container), a program, a function, or a node holding a model."""

    fn: Any
    args: tuple[Any, ...]
    kwargs: dict[str, Any] = field(default_factory=dict)

    @property
    def op(self) -> Any:
        """What is called, whatever its settings: a config's method, or fn itself."""
        if isinstance(self.fn, Method):
            return type(self.fn)
        if isinstance(self.fn, Call) and _makes_config(self.fn):   # a config made in the tree, then called
            return self.fn.fn
        return self.fn

    @property
    def runs(self) -> bool:
        """Whether it runs a method (a config called on data), rather than making something."""
        return isinstance(self.fn, Method) or (isinstance(self.fn, Call) and _makes_config(self.fn))

    @property
    def settings(self) -> dict[str, Any]:
        """Its config's settings that differ from their defaults, and the keywords it is called with."""
        return (settings_of(self.fn) if isinstance(self.fn, Method) else {}) | self.kwargs

    def with_settings(self, **settings: Any) -> Call:
        """The same call with some settings changed: in its config, or in the keywords it is called with."""
        if isinstance(self.fn, Method) and is_dataclass(self.fn):
            return replace(self, fn=replace(self.fn, **settings))
        return replace(self, kwargs={**self.kwargs, **settings})


def _makes_config(c: Call) -> bool:
    return isinstance(c.fn, type) and issubclass(c.fn, Method)


def inputs(names: str) -> tuple[Input, ...]:
    return tuple(Input(n) for n in names.split())


def call(fn: Any, *args: Any, **kwargs: Any) -> Call:
    """fn applied to nodes (or constants): call(Walk(steps=8), tree, Greedy())."""
    return Call(fn, args, kwargs)


def calls(op: Any) -> Callable[[Any], bool]:
    """Matches the calls of a method (whatever its settings), or of anything else called: calls(Walk)."""
    return lambda e: isinstance(e, Call) and e.op is op


def _children(e: Any) -> list[Any]:
    if isinstance(e, Call):
        return [e.fn, *e.args, *e.kwargs.values()]
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
        if not isinstance(e, Node):
            return e
        if id(e) not in memo:
            if isinstance(e, Input):
                if e.name not in env:
                    raise NameError(f"{where}: {e.name} is not bound")
                memo[id(e)] = env[e.name]
            elif isinstance(e, Attr):
                memo[id(e)] = _reach(run(e.base), e.attr)
            elif isinstance(e, Call):
                memo[id(e)] = run(e.fn)(*map(run, e.args), **{k: run(v) for k, v in e.kwargs.items()})
        return memo[id(e)]

    return run(root)


def _nodes(roots: list[Any]) -> list[Node]:
    """Every node below the roots, children before parents, each once."""
    seen: dict[int, Node] = {}

    def visit(e: Any) -> None:
        if isinstance(e, Node) and id(e) not in seen:
            for c in _children(e):
                visit(c)
            seen[id(e)] = e

    for r in roots:
        visit(r)
    return list(seen.values())


def _constants(e: Call) -> list[Any]:
    """What a call holds that is not a node: its function, its constant arguments, its config's settings."""
    config = [getattr(e.fn, f.name) for f in fields(e.fn)] if isinstance(e.fn, Method) and is_dataclass(e.fn) else []
    return [e.fn, *e.args, *e.kwargs.values(), *config]


def _rewriter(match: Callable[[Any], bool], change: Callable[[Any], Any]) -> Callable[[Any], Any]:
    """Rewrites expressions bottom-up, every matching node replaced by change(node); shared nodes stay shared (one
    memo for every expression it is given), and programs used as constants (losses) are rewritten too."""
    memo: dict[int, Any] = {}

    def go(e: Any) -> Any:
        if isinstance(e, (Program, Script)):
            return e.rewrite(match, change)
        if not isinstance(e, Node):
            return e
        if id(e) not in memo:
            new = e
            if isinstance(e, Call):
                fn, args, kwargs = go(e.fn), tuple(go(a) for a in e.args), {k: go(v) for k, v in e.kwargs.items()}
                if fn is not e.fn or any(a is not b for a, b in zip(args, e.args)) or any(kwargs[k] is not v for k, v in e.kwargs.items()):
                    new = replace(e, fn=fn, args=args, kwargs=kwargs)
            elif isinstance(e, Attr) and (base := go(e.base)) is not e.base:
                new = replace(e, base=base)
            memo[id(e)] = change(new) if match(new) else new
        return memo[id(e)]

    return go


def _misplaced(where: str, root: Any, top: bool, update: Any = None) -> list[str]:
    """Statements where they don't belong. With top, root is a statement's own expression: if it runs a statement,
    that call and the dataset statements its arguments nest are at the top; every other statement run is inside an
    expression. A call that changes its argument belongs only at the top of an update (update)."""
    at_top: set[int] = set()

    def statement(e: Any) -> bool:
        return isinstance(e, Call) and e.runs and is_statement(e.op)

    def mark(e: Any) -> None:
        if statement(e) and id(e) not in at_top:
            at_top.add(id(e))
            for a in (*e.args, *e.kwargs.values()):
                mark(a)

    if top:
        mark(root)
    out = []
    for e in _nodes([root]):
        if statement(e):
            assert isinstance(e, Call)
            if id(e) not in at_top:
                out.append(f"{where}: {_name(e.op)} is a statement inside an expression; make it a line of its own (let, update)")
            elif changes(e.op) is not None and e is not update:
                out.append(f"{where}: {_name(e.op)} changes its argument; make it an update")
    return out


def _sub_checks(roots: list[Any]) -> list[str]:
    subs = {id(a): a for e in _nodes(roots) if isinstance(e, Call) for a in _constants(e) if isinstance(a, (Program, Script))}
    return [c for sub in subs.values() for c in sub.check()]


# ═══ Program: an expression ═══════════════════════════════════════════════════════════════════════


@dataclass(frozen=True, eq=False)
class Program:
    """A named pure expression with inputs: calling it evaluates the tree, each shared node once."""

    name: str
    inputs: tuple[Input, ...]
    output: Node
    about: str = ""

    def __call__(self, *args: Any) -> Any:
        if len(args) != len(self.inputs):
            raise TypeError(f"{self.name} takes ({', '.join(i.name for i in self.inputs)}), got {len(args)} values")
        return _evaluate(self.output, {i.name: a for i, a in zip(self.inputs, args)}, self.name)

    def nodes(self) -> list[Node]:
        return _nodes([self.output])

    def find(self, match: Callable[[Any], bool]) -> list[Node]:
        return [e for e in self.nodes() if match(e)]

    def rewrite(self, match: Callable[[Any], bool], change: Callable[[Any], Any]) -> Program:
        """A new program with every matching node replaced by change(node); this one when nothing matched, so that
        what holds it (a Train given it as its loss) is unchanged too."""
        output = _rewriter(match, change)(self.output)
        return self if output is self.output else replace(self, output=output)

    def check(self) -> list[str]:
        """An expression runs no statement anywhere in it, nor in the programs it uses."""
        return _misplaced(self.name, self.output, top=False) + _sub_checks([self.output])

    def __str__(self) -> str:
        lines = [f"{self.name}({', '.join(i.name for i in self.inputs)}) → {_text(self.output)}"] + ([f"  {self.about}"] if self.about else [])
        return "\n".join(lines + _columns(_tree(self.output, "")) + _where([self.output]))


# ═══ Script: statements ═══════════════════════════════════════════════════════════════════════════


@dataclass(frozen=True)
class Let:
    """name = value: a name for the value of an expression, or of a dataset statement."""

    name: Input
    value: Any


@dataclass(frozen=True)
class Update:
    """op(…, name!, …): an update statement, changing the named value in place (it returns None)."""

    name: Input
    call: Call


@dataclass(frozen=True)
class Repeat:
    """The statements, in order, times times; with until, an expression, at most times times, stopping as soon as it
    is true (it is read before each pass)."""

    times: int
    body: tuple[Statement, ...]
    until: Any = None


type Statement = Let | Update | Repeat


def let(name: Input, value: Any) -> Let:
    return Let(name, value)


def update(name: Input, op: Any, *args: Any) -> Update:
    """op(name, *args): an update changes its first argument, name, in place. op is an update's config, or a call
    that makes one."""
    i = changes(op.op if isinstance(op, Call) else op)
    if i is None:
        raise TypeError(f"{_name(op)} changes nothing; an update needs a command that does (TrainCmd, DeployCmd, IndexCmd, …)")
    return Update(name, call(op, *args[:i], name, *args[i:]))


def repeat(times: int, *body: Statement, until: Any = None) -> Repeat:
    return Repeat(times, body, until)


def _flat(body: tuple[Statement, ...]) -> Iterator[Let | Update]:
    for s in body:
        yield from _flat(s.body) if isinstance(s, Repeat) else (s,)


def _untils(body: tuple[Statement, ...]) -> Iterator[Any]:
    for s in body:
        if isinstance(s, Repeat):
            yield from (s.until,) if s.until is not None else ()
            yield from _untils(s.body)


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
                        if s.until is not None and _evaluate(s.until, env, f"{self.name}: until"):
                            break
                        run(s.body)
                elif isinstance(s, Let):
                    env[s.name.name] = _evaluate(s.value, env, f"{self.name}: {s.name.name}")
                else:  # the arguments first, then the update, in place: the name still names the same value
                    _evaluate(s.call, env, f"{self.name}: {s.name.name}")

        run(self.body)
        return _evaluate(self.output, env, self.name)

    def _roots(self) -> list[Any]:
        return [*map(_expr, _flat(self.body)), *_untils(self.body), self.output]

    def nodes(self) -> list[Node]:
        return _nodes(self._roots())

    def find(self, match: Callable[[Any], bool]) -> list[Node]:
        return [e for e in self.nodes() if match(e)]

    def rewrite(self, match: Callable[[Any], bool], change: Callable[[Any], Any]) -> Script:
        go = _rewriter(match, change)

        def statement(s: Statement) -> Statement:
            if isinstance(s, Repeat):
                body, until = statements(s.body), go(s.until)
                return s if body is s.body and until is s.until else Repeat(s.times, body, until)
            if isinstance(s, Let):
                value = go(s.value)
                return s if value is s.value else Let(s.name, value)
            c = go(s.call)
            return s if c is s.call else Update(s.name, c)

        def statements(body: tuple[Statement, ...]) -> tuple[Statement, ...]:
            new = tuple(map(statement, body))
            return body if all(a is b for a, b in zip(new, body)) else new

        body, output = statements(self.body), go(self.output)
        return self if body is self.body and output is self.output else replace(self, body=body, output=output)

    def check(self) -> list[str]:
        """Statements only at the top of a line, a change only as an update of its own name; every name bound before
        it is read."""
        out: list[str] = []
        bound = {i.name for i in self.inputs}

        def unbound(where: str, e: Any) -> list[str]:
            return [f"{where}: reads {n.name} before it is bound" for n in _nodes([e]) if isinstance(n, Input) and n.name not in bound]

        def walk(body: tuple[Statement, ...]) -> None:
            for s in body:
                if isinstance(s, Repeat):
                    if s.until is not None:   # an expression, read before the first pass
                        out.extend(_misplaced(f"{self.name}: until", s.until, top=False) + unbound(f"{self.name}: until", s.until))
                    walk(s.body)
                    continue
                where = f"{self.name}: {s.name.name}"
                if isinstance(s, Update):
                    i = changes(s.call.op)
                    if not s.call.runs or i is None or i >= len(s.call.args) or s.call.args[i] is not s.name:
                        out.append(f"{where}: {_name(s.call.op)} is not an update of {s.name.name}")
                out.extend(_misplaced(where, _expr(s), top=True, update=s.call if isinstance(s, Update) else None) + unbound(where, _expr(s)))
                bound.add(s.name.name)

        walk(self.body)
        out += _misplaced(self.name, self.output, top=False)
        out += [f"{self.name}: reads {e.name}, never bound" for e in _nodes([self.output]) if isinstance(e, Input) and e.name not in bound]
        return out + _sub_checks(self._roots())

    def __str__(self) -> str:
        rows: list[tuple[str, str]] = []

        def show(body: tuple[Statement, ...], depth: int) -> None:
            pad = "  " * depth
            for s in body:
                if isinstance(s, Repeat):
                    head = _tree(s.until, pad, f"repeat ×{s.times}, until ") if s.until is not None else [(f"    {pad}repeat ×{s.times}", "")]
                    rows.extend([(head[0][0] + ":", head[0][1]), *head[1:]])
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


def _name(op: Any) -> str:
    """What a call is called: a method or a class by its name, a program by its own, a call by what it calls."""
    if isinstance(op, Call):
        return _name(op.op)
    if isinstance(op, Method):
        return _name(type(op))
    if isinstance(op, type) and issubclass(op, Intent):
        return op.__name__ + "?"                     # an intent: some method of its group, not chosen yet
    return op.name if isinstance(op, (Program, Script)) else getattr(op, "__name__", repr(op))


def _short(v: Any) -> str:
    """A constant as an argument: a program by its name, a config as Beam(k=4), a function or a class by its name."""
    if isinstance(v, (Program, Script)):
        return v.name
    if isinstance(v, Intent):
        return _name(v)
    if isinstance(v, Method):
        return short(v)
    return getattr(v, "__name__", None) or repr(v)


def _text(e: Any) -> str:
    """A node as an argument: its label, or a short form."""
    if not isinstance(e, Node):
        return _short(e)
    if isinstance(e, Input):
        return e.name
    if isinstance(e, Attr):
        return e.label or _text(e.base) + _attr(e.attr)
    assert isinstance(e, Call)
    return e.label or (_text(e.fn) if isinstance(e.fn, Node) and not _makes_config_node(e.fn) else _name(e)) + "(…)"


def _attr(attr: str) -> str:
    return f"[{attr}]" if attr.lstrip("-").isdigit() else f".{attr}"


def _makes_config_node(n: Node) -> bool:
    return isinstance(n, Call) and _makes_config(n)


def _tree(root: Any, pad: str, lhs: str = "", changes: Input | None = None) -> list[tuple[str, str]]:
    """Rows (text, group) for the calls of an expression, from its root down, each shared call once: a call shows
    what it calls, its arguments, then its settings after a ";". changes is the name an update changes, marked !."""
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
        made = e.fn if isinstance(e.fn, Call) and _makes_config(e.fn) else None   # a config made in the tree, then called
        name = _text(e.fn) if isinstance(e.fn, Node) and made is None else _name(e)
        args = ", ".join(_text(a) + ("!" if depth == 0 and a is changes else "") for a in e.args)
        settings = made.kwargs if made is not None else e.settings              # its settings shown here, with the call
        shown = ", ".join(f"{k}={_text(v)}" for k, v in settings.items())
        rows.append((f"    {pad}{'  ' * depth}{lhs}{name}({args}{'; ' if args and shown else ''}{shown})", group_of(e)))
        for c in calls_below(e):
            if c is not made:
                show(c, depth + 1, f"{c.label} = " if c.label else "")

    top, path = root, ""
    while isinstance(top, Attr):
        top, path = top.base, _attr(top.attr) + path
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
    subs = {id(a): a for e in _nodes(roots) if isinstance(e, Call) for a in _constants(e) if isinstance(a, (Program, Script))}
    return [line for sub in subs.values() for line in [f"  where {sub.name} is", *("    " + l for l in str(sub).splitlines())]]


def group_of(e: Call) -> str:
    """The catalog group of a call: "model" for a node holding a model, "program" for a program, "·" for glue."""
    if isinstance(e.fn, Node) and not e.runs:
        return "model"
    op = e.op
    if isinstance(op, (Program, Script)) or op in BUILDERS:
        return "program"
    return group_class(op).group if isinstance(op, type) and issubclass(op, Method) else "·"


def groups(p: Program | Script) -> list[str]:
    """The groups a program or script draws on, its sub-programs' included, in order of first use."""
    seen: list[str] = []
    for e in p.nodes():
        if isinstance(e, Call):
            g = group_of(e)
            if g not in seen and g not in ("model", "program", "·"):
                seen.append(g)
            for a in _constants(e):
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


def taken(values: Any, action: Any) -> Any:
    """The value of the action taken, out of one per action."""
    return values.gather(-1, action.unsqueeze(-1)).squeeze(-1)


def q_target(q: Callable[[Any], Any], batch: Any, gamma: float = 0.99) -> Any:
    """r + γ·max Q(s′), nothing past the end: what the action taken learns toward (no gradient through it)."""
    return (batch.reward + gamma * (1 - batch.done.float()) * q(batch.next_state).max(-1).values).detach()


def search_targets(path: Path, search: PartialTree[Any, Any]) -> tuple[Any, Any]:
    """AlphaZero's: for each state a walk visited, what search made of it (its visit distribution) and what followed (the
    return)."""
    ...


# ═══ Losses, as programs: a split, a model, a score ═══════════════════════════════════════════════


def split_loss(name: str, split: Callable[..., Example], about: str) -> Program:
    """A loss of (model, batch): split the batch, predict from the input, score with the loss the target picks.

        def split_loss(net: Predictor, data: Container, *, split: Callable[..., Example]) -> Tensor:
            ex: Example = split(data)
            return Loss()(net(ex.input), ex.target, ex.weight)
    """
    model, data = inputs("network data")
    ex = call(split, data).named("ex")
    return Program(name, (model, data), call(Loss(), model(ex.input), ex.target, ex.weight), about)


next_token_loss = split_loss("next-token loss", NextToken(), "each token predicted from the ones before it")
masked_loss = split_loss("masked loss", Mask(), "random holes, predicted")
autoencoding_loss = split_loss("reconstruction loss", Reconstruction(), "the input as its own target, through the bottleneck")


def _supervised_loss() -> Program:
    model, x, y = inputs("network x y")
    return Program("supervised loss", (model, x, y), call(Loss(), model(x), y), "the label as the target")


def _dynamics_loss() -> Program:
    world, batch = inputs("world batch")
    return Program("dynamics loss", (world, batch), call(Loss(), world(call(Condition(), batch.state, batch.action)), batch.next_state),
                   "from a state and an action, what follows")


def _flow_matching_loss() -> Program:
    velocity, z, labels = inputs("velocity z labels")
    ex = call(Noise(), z).named("ex")
    return Program("conditional flow-matching loss", (velocity, z, labels), call(Loss(), velocity(ex.input, labels), ex.target),
                   "noised latents; the velocity back, now and then without the label")


def _contrastive_loss() -> Program:
    pair, images, texts = inputs("towers images texts")
    return Program("contrastive loss", (pair, images, texts), call(InfoNCE(), pair.image(images), pair.text(texts)),
                   "which caption goes with which image, among the batch")


def _ppo_loss() -> Program:
    agent, batch = inputs("agent batch")
    logits, value = agent.policy(batch.observation).named("logits"), agent.critic(batch.observation).named("value")
    advantage = call(operator.sub, batch.returns, value).named("advantage")
    policy_loss = call(PPO(), call(LogProb(), logits, batch.action), batch.log_prob, advantage).named("policy_loss")
    return Program("PPO loss", (agent, batch), call(operator.add, policy_loss, call(Loss(), value, batch.returns)),
                   "the clipped policy gradient with the critic as baseline, plus the critic's regression")


def _q_learning_loss() -> Program:
    q, batch = inputs("q batch")
    return Program("Q-learning loss", (q, batch), call(Loss(), call(taken, q(batch.state), batch.action), call(q_target, q, batch)),
                   "the action taken learns toward r + γ·max Q(s′)")


supervised_loss, dynamics_loss, flow_matching_loss, contrastive_loss, ppo_loss, q_learning_loss = (
    _supervised_loss(), _dynamics_loss(), _flow_matching_loss(), _contrastive_loss(), _ppo_loss(), _q_learning_loss())


def grpo_loss(verify: Callable[..., Any], n: int = 8, length: int = 1) -> Program:
    """The same thing in plain Python:

        def grpo_loss(policy: Predictor, prompts: Temporal, *, verify: Callable[[Any], Value], n: int = 8,
                      length: int = 1) -> Tensor:
            path: Path = Walk(steps=length)(LMTree()(policy, prompts), NPaths(n=n, choose=Sample()))
            return Reinforce()(path, GroupAdvantage(n=n)(verify(path.end)))
    """
    policy, prompts = inputs("policy prompts")
    path = call(Walk(steps=length), call(LMTree(), policy, prompts), NPaths(n=n, choose=Sample())).named("path")
    advantage = call(GroupAdvantage(n=n), call(verify, path.end))
    return Program("GRPO loss", (policy, prompts), call(Reinforce(), path, advantage),
                   "n answers per prompt, verified, each pushed by how it did against its group")


BUILDERS.add(grpo_loss)

# ═══ Well-known work ══════════════════════════════════════════════════════════════════════════════
# Training is a statement, so anything that trains is a script; the losses above, and what only reads a model, are
# programs. A corpus is a dataset (its __domain__ is what its batches are, read without one); what fits in memory is made one by
# Batches.


def language_model(name: str = "GPT", about: str = "a causal Transformer: shift, the loss, train; then walk the LM's tree",
                   mixer: LayerOp[Any, Any] | None = None, mlp: LayerOp[Any, Any] | None = None, steps: int = 200) -> Script:
    """The same thing in plain Python (name and about are only labels):

        def language_model(corpus: Dataset[Temporal], prompt: Temporal, *, mixer: LayerOp | None = None,
                           mlp: LayerOp | None = None, steps: int = 200) -> Temporal:
            first: Temporal = corpus.__domain__
            ex: Example = NextToken()(first)
            net: Predictor = Network(target=ex.target, mixer=mixer, mlp=mlp)
            Train(steps=steps)(net, next_token_loss, corpus)
            return Walk(steps=8)(LMTree()(net, prompt), Greedy()).end
    """
    corpus, prompt, net, first = inputs("corpus prompt network first")
    ex = call(NextToken(), first).named("ex")
    return Script(name, (corpus, prompt), (
        let(first, Attr(corpus, "__domain__")),
        let(net, call(Network, target=ex.target, mixer=mixer, mlp=mlp)),
        update(net, Train(steps=steps), next_token_loss, corpus),
    ), call(Walk(steps=8), call(LMTree(), net, prompt), Greedy()).end, about)


def mamba() -> Script:
    """language_model with Network(mixer=SSM(d=64))."""
    return language_model("Mamba", "GPT with a selective scan in place of attention", mixer=SSM(d=64))


def mixtral() -> Script:
    """language_model with Network(mlp=MoE(d=64))."""
    return language_model("Mixtral / DeepSeek-V3", "GPT whose MLP is a routed mixture of experts", mlp=MoE(d=64))


def vit(steps: int = 100) -> Script:
    """The same thing in plain Python:

        def vit(images: Grid, labels: Value, *, steps: int = 100) -> Tensor:
            tokens: Seq = Flatten()(Patchify(size=4)(images))
            net: Predictor = Network(target=labels)
            Train(steps=steps)(net, supervised_loss, Batches(size=64)(tokens, labels))
            return Accuracy()(net(tokens), labels)
    """
    images, labels, tokens, net = inputs("images labels tokens network")
    return Script("ViT", (images, labels), (
        let(tokens, call(Flatten(), call(Patchify(size=4), images))),
        let(net, call(Network, target=labels)),
        update(net, Train(steps=steps), supervised_loss, call(Batches(size=64), tokens, labels)),
    ), call(Accuracy(), net(tokens), labels), "an image as a sequence of patches, read by a Transformer")


def clip(dim: int = 16, steps: int = 150) -> Script:
    """The same thing in plain Python:

        def clip(images: Grid, texts: Temporal, *, dim: int = 16, steps: int = 150) -> Tensor:
            patches: Grid = Patchify(size=4)(images)
            image: Model[Container, Value] = Encoder(dim=dim)
            text: Model[Container, Value] = Encoder(dim=dim)
            pair: nn.ModuleDict = towers(image, text)
            Train(steps=steps)(pair, contrastive_loss, Batches(size=256)(patches, texts))
            return RetrievalAccuracy()(pair.image(patches), pair.text(texts))
    """
    images, texts, patches, image, text, pair = inputs("images texts patches image text towers")
    return Script("CLIP", (images, texts), (
        let(patches, call(Patchify(size=4), images)),
        let(image, call(Encoder, dim=dim)),
        let(text, call(Encoder, dim=dim)),
        let(pair, call(towers, image, text)),
        update(pair, Train(steps=steps), contrastive_loss, call(Batches(size=256), patches, texts)),
    ), call(RetrievalAccuracy(), pair.image(patches), pair.text(texts)),
        "an image encoder and a text encoder, trained so that each image picks out its caption among its batch")


def stable_diffusion(latent: int = 4, steps: int = 150, w: float = 3.0) -> Script:
    """The same thing in plain Python:

        def stable_diffusion(images: Grid, labels: Value, *, latent: int = 4, steps: int = 150,
                             w: float = 3.0) -> Grid:
            ae: Model[Container, Container] = Autoencoder(latent=latent)
            Train(steps=steps)(ae, autoencoding_loss, Batches(size=64)(images))
            z: Container = ae.encode(images)
            ex: Example = Noise()(z)
            net: Predictor = Network(target=ex.target)
            velocity: Predictor = CFGDropout(p=0.1)(net)
            Train(steps=steps)(velocity, flow_matching_loss, Batches(size=64)(z, labels))
            sampler: Predictor = CFG(w=w)(Given()(velocity, labels), Given()(velocity, None))
            return ae.decode(Fill(steps=10)(Blank()(z), sampler, AllAtOnce()))
    """
    images, labels, ae, z, net, velocity = inputs("images labels ae z network velocity")
    ex = call(Noise(), z).named("ex")
    sampler = call(CFG(w=w), call(Given(), velocity, labels), call(Given(), velocity, None))
    return Script("Stable Diffusion", (images, labels), (
        let(ae, call(Autoencoder, latent=latent)),
        update(ae, Train(steps=steps), autoencoding_loss, call(Batches(size=64), images)),
        let(z, ae.encode(images)),
        let(net, call(Network, target=ex.target)),
        let(velocity, call(CFGDropout(p=0.1), net)),
        update(velocity, Train(steps=steps), flow_matching_loss, call(Batches(size=64), z, labels)),
    ), ae.decode(call(Fill(steps=10), call(Blank(), z), sampler, AllAtOnce())),
        "flow matching in an autoencoder's latent space, with a label, sampled with classifier-free guidance")


def lora_finetune(rank: int = 4, steps: int = 150) -> Script:
    """The same thing in plain Python:

        def lora_finetune(net: Predictor, corpus: Dataset[Temporal], prompt: Temporal, *, rank: int = 4,
                          steps: int = 150) -> Temporal:
            LoRA(rank=rank)(net)
            Train(steps=steps)(net, next_token_loss, corpus)
            return Walk(steps=8)(LMTree()(net, prompt), Greedy()).end
    """
    net, corpus, prompt = inputs("network corpus prompt")
    return Script("LoRA finetuning", (net, corpus, prompt), (
        update(net, LoRA(rank=rank)),
        update(net, Train(steps=steps), next_token_loss, corpus),
    ), call(Walk(steps=8), call(LMTree(), net, prompt), Greedy()).end, "a pretrained model, frozen, with a trainable low-rank update")


def dpo(beta: float = 0.1, steps: int = 60) -> Script:
    """The same thing in plain Python:

        def dpo(policy: Predictor, chosen: Temporal, rejected: Temporal, *, beta: float = 0.1,
                steps: int = 60) -> Predictor:
            reference: Predictor = FrozenCopy()(policy)
            Train(steps=steps)(policy, DPO(reference=reference, beta=beta), Batches(size=8)(chosen, rejected))
            return policy
    """
    policy, chosen, rejected, reference = inputs("policy chosen rejected reference")
    return Script("DPO", (policy, chosen, rejected), (
        let(reference, call(FrozenCopy(), policy)),
        update(policy, Train(steps=steps), call(DPO, reference=reference, beta=beta), call(Batches(size=8), chosen, rejected)),
    ), policy, "preference pairs as a classification of log-ratios against a frozen reference")


def alphazero(rounds: int = 8, simulations: int = 32, steps: int = 30, length: int = 16) -> Script:
    """The same thing in plain Python:

        def alphazero(net: Predictor, game: Callable[[Predictor], PartialTree[Any, Tensor]], *, rounds: int = 8,
                      simulations: int = 32, steps: int = 30, length: int = 16) -> Path:
            for _ in range(rounds):
                search = MCTS(simulations=simulations, two_player=True)(game(net))
                states, targets = search_targets(Walk(steps=length)(search, Sample()), search)
                Train(steps=steps)(net, supervised_loss, Batches(size=64)(states, targets))
            return Walk(steps=length)(game(net), Greedy())
    """
    net, game, ex = inputs("network game targets")
    search = call(MCTS(simulations=simulations, two_player=True), game(net)).named("search")
    return Script("AlphaZero", (net, game), (
        repeat(rounds,
               let(ex, call(search_targets, call(Walk(steps=length), search, Sample()), search)),
               update(net, Train(steps=steps), supervised_loss, call(Batches(size=64), ex[0], ex[1]))),
    ), call(Walk(steps=length), game(net), Greedy()),
        "a policy-value network that amortizes MCTS: each round plays itself with search, then learns the search's choices")


def r1(n: int = 8, length: int = 1, steps: int = 100) -> Script:
    """The same thing in plain Python:

        def r1(policy: Predictor, prompts: Temporal, verify: Callable[[Any], Value], *, n: int = 8, length: int = 1,
               steps: int = 100) -> Value:
            Train(steps=steps)(policy, grpo_loss(verify, n=n, length=length), Batches(size=8)(prompts))
            return verify(Walk(steps=length)(LMTree()(policy, prompts), Greedy()).end)
    """
    policy, prompts, verify = inputs("policy prompts verify")
    return Script("DeepSeek-R1 (GRPO)", (policy, prompts, verify), (
        update(policy, Train(steps=steps), call(grpo_loss, verify, n=n, length=length), call(Batches(size=8), prompts)),
    ), verify(call(Walk(steps=length), call(LMTree(), policy, prompts), Greedy()).end),
        "RL on checkable answers: walk, verify, group-relative advantages (each group in one batch), REINFORCE")


def diffusion_lm(steps: int = 600) -> Script:
    """The same thing in plain Python:

        def diffusion_lm(corpus: Dataset[Temporal], *, steps: int = 600) -> Temporal:
            first: Temporal = corpus.__domain__
            ex: Example = Mask()(first)
            net: Predictor = Network(target=ex.target)
            Train(steps=steps)(net, masked_loss, corpus)
            return Fill(steps=8)(Blank()(first), net, MostConfident())
    """
    corpus, net, first = inputs("corpus network first")
    ex = call(Mask(), first).named("ex")
    return Script("Diffusion LLM", (corpus,), (
        let(first, Attr(corpus, "__domain__")),
        let(net, call(Network, target=ex.target)),
        update(net, Train(steps=steps), masked_loss, corpus),
    ), call(Fill(steps=8), call(Blank(), first), net, MostConfident()),
        "masked prediction at every ratio; generation unmasks the surest holes first")


# ═══ Compositions the catalog points to ═══════════════════════════════════════════════════════════


def pretrain_finetune(steps: int = 150) -> Script:
    """The same thing in plain Python:

        def pretrain_finetune(corpus: Dataset[Temporal], texts: Temporal, labels: Value, *, steps: int = 150) -> Predictor:
            first: Temporal = corpus.__domain__
            ex: Example = NextToken()(first)
            lm: Predictor = Network(target=ex.target)
            Train(steps=steps)(lm, next_token_loss, corpus)
            clf: Predictor = Network(target=labels)
            Transfer()(clf, lm)
            Freeze()(clf, head)
            Train(steps=steps)(clf, supervised_loss, Batches(size=32)(texts, labels))
            return clf
    """
    corpus, texts, labels, first, lm, clf = inputs("corpus texts labels first lm classifier")
    ex = call(NextToken(), first).named("ex")
    return Script("Pretrain → finetune", (corpus, texts, labels), (
        let(first, Attr(corpus, "__domain__")),
        let(lm, call(Network, target=ex.target)),
        update(lm, Train(steps=steps), next_token_loss, corpus),
        let(clf, call(Network, target=labels)),
        update(clf, Transfer(), lm),
        update(clf, Freeze(), head),
        update(clf, Train(steps=steps), supervised_loss, call(Batches(size=32), texts, labels)),
    ), clf, "learn from raw text, then reuse the body under a new head")


def ppo(rounds: int = 30, horizon: int = 20, epochs: int = 4) -> Script:
    """The same thing in plain Python:

        def ppo(agent: nn.ModuleDict, env: Environment, *, rounds: int = 30, horizon: int = 20,
                epochs: int = 4) -> Value:
            for _ in range(rounds):
                path: Path = Walk(steps=horizon)(EnvTree()(env, agent.policy), Sample())
                batch: Value = Transitions()(path, ReturnsToGo(gamma=0.9)(path.rewards))
                Train(steps=epochs)(agent, ppo_loss, Batches(size=64)(batch))
            return DiscountedReturn()(Walk(steps=horizon)(EnvTree()(env, agent.policy), Greedy()))
    """
    agent, env, batch = inputs("agent env batch")
    path = call(Walk(steps=horizon), call(EnvTree(), env, agent.policy), Sample()).named("path")
    return Script("PPO (actor-critic)", (agent, env), (
        repeat(rounds,
               let(batch, call(Transitions(), path, call(ReturnsToGo(gamma=0.9), path.rewards))),
               update(agent, Train(steps=epochs), ppo_loss, call(Batches(size=64), batch))),
    ), call(DiscountedReturn(), call(Walk(steps=horizon), call(EnvTree(), env, agent.policy), Greedy())),
        "a policy and a critic; each round walks the environment, then takes a few clipped steps on what was seen")


def dqn(rounds: int = 30, horizon: int = 20, steps: int = 10, eps: float = 0.3) -> Script:
    """The same thing in plain Python:

        def dqn(q: Predictor, env: Environment, *, rounds: int = 30, horizon: int = 20, steps: int = 10,
                eps: float = 0.3) -> Value:
            buffer: Buffer[Value] = ReplayBuffer(size=10_000)()
            for _ in range(rounds):
                path: Path = Walk(steps=horizon)(EpsilonGreedy(eps=eps)(EnvTree()(env, q)), Sample())
                Extend()(buffer, Transitions()(path))
                Train(steps=steps)(q, q_learning_loss, buffer)
            return DiscountedReturn()(Walk(steps=horizon)(EnvTree()(env, q), Greedy()))
    """
    q, env, buffer = inputs("q env buffer")
    walked = call(Walk(steps=horizon), call(EpsilonGreedy(eps=eps), call(EnvTree(), env, q)), Sample())
    return Script("DQN", (q, env), (
        let(buffer, call(ReplayBuffer(size=10_000))),
        repeat(rounds,
               update(buffer, Extend(), call(Transitions(), walked)),
               update(q, Train(steps=steps), q_learning_loss, buffer)),
    ), call(DiscountedReturn(), call(Walk(steps=horizon), call(EnvTree(), env, q), Greedy())),
        "values learned by bootstrapping: each round walks ε-greedily into a replay buffer, then regresses Q on samples of it")


def world_model_planning(horizon: int = 20, steps: int = 300) -> Script:
    """The same thing in plain Python:

        def world_model_planning(env: Environment, *, horizon: int = 20, steps: int = 300) -> Value:
            tree: PartialTree[Any, Tensor] = EnvTree()(env)
            batch: Value = Transitions()(Walk(steps=horizon)(tree, Sample()))
            world: Predictor = Network(target=batch.next_state)
            Train(steps=steps)(world, dynamics_loss, Batches(size=64)(batch))
            plan = MPC(horizon=8, samples=32)(tree, WorldModel()(tree, world))
            return DiscountedReturn()(Walk(steps=horizon)(plan, Greedy()))
    """
    env, batch, world = inputs("env batch world")
    tree = call(EnvTree(), env).named("tree")
    plan = call(MPC(horizon=8, samples=32), tree, call(WorldModel(), tree, world)).named("plan")
    return Script("Planning in a learned world model", (env,), (
        let(batch, call(Transitions(), call(Walk(steps=horizon), tree, Sample()))),
        let(world, call(Network, target=batch.next_state)),
        update(world, Train(steps=steps), dynamics_loss, call(Batches(size=64), batch)),
    ), call(DiscountedReturn(), call(Walk(steps=horizon), plan, Greedy())),
        "explore at random, learn what follows each action, act by planning inside that model (Dreamer, MuZero)")


def synthetic_data(n: int = 8, length: int = 4, threshold: float = 0.5) -> Program:
    """The same thing in plain Python (no statement in it, so a Program):

        def synthetic_data(generator: Predictor, prompts: Temporal, verify: Callable[[Any], Value], *, n: int = 8,
                           length: int = 4, threshold: float = 0.5) -> Temporal:
            tree = Temperature(t=1.5)(LMTree()(generator, prompts))
            path: Path = Walk(steps=length)(tree, NPaths(n=n, choose=Sample()))
            return QualityFilter(threshold=threshold)(Temporal(path.end), verify(path.end))
    """
    generator, prompts, verify = inputs("generator prompts verify")
    tree = call(Temperature(t=1.5), call(LMTree(), generator, prompts))
    path = call(Walk(steps=length), tree, NPaths(n=n, choose=Sample())).named("path")
    return Program("Synthetic data", (generator, prompts, verify),
                   call(QualityFilter(threshold=threshold), call(Temporal, path.end), verify(path.end)),
                   "sample a model's tree widely, keep what a verifier scores above a threshold")


def curation(keep: float = 0.5) -> Script:
    """The same thing in plain Python (every line a dataset statement):

        def curation(web: Indexed[Temporal], books: Indexed[Temporal], code: Indexed[Temporal], quality: Callable[[Temporal], Value],
                     difficulty: Callable[[Temporal], Value], *, keep: float = 0.5) -> Subset[Temporal]:
            mixed: Indexed[Temporal] = Mix(weights=[0.7, 0.2, 0.1])(web, books, code)
            kept: Subset[Temporal] = KeepBest(keep=keep)(NearDedup()(Dedup()(mixed)), quality)
            return Curriculum()(kept, difficulty)       # a Subset of mixed: it remembers each step that chose it
    """
    web, books, code, quality, difficulty, data = inputs("web books code quality difficulty data")
    mixed = call(Mix(weights=[0.7, 0.2, 0.1]), web, books, code)
    kept = call(KeepBest(keep=keep), call(NearDedup(), call(Dedup(), mixed)), quality)
    return Script("Data curation", (web, books, code, quality, difficulty), (
        let(data, call(Curriculum(), kept, difficulty)),
    ), data, "mix the sources, drop copies and near-copies, keep the better half, easy first: every step reads every batch")


def early_stopping(steps: int = 5000, every: int = 100, patience: int = 3) -> Script:
    """The same thing in plain Python:

        def early_stopping(images: Grid, labels: Value, *, steps: int = 5000, every: int = 100,
                           patience: int = 3) -> Tensor:
            splits: tuple[tuple[Subset[Any], ...], ...] = TrainTestSplit(fractions=(0.8, 0.1, 0.1))(Batches(size=64)(images, labels))
            net: Predictor = Network(target=labels)
            Train(steps=steps, patience=patience, every=every)(net, supervised_loss, splits[0][0], splits[0][1])
            test: Tensor = Evaluate()(splits[0][2], supervised_loss, net)
            return test
    """
    images, labels, splits, net, test = inputs("images labels splits network test")
    train, validation, held_out = splits[0][0], splits[0][1], splits[0][2]
    return Script("Early stopping", (images, labels), (
        let(splits, call(TrainTestSplit(fractions=(0.8, 0.1, 0.1)), call(Batches(size=64), images, labels))),
        let(net, call(Network, target=labels)),
        update(net, Train(steps=steps, patience=patience, every=every), supervised_loss, train, validation),
        let(test, call(Evaluate(), held_out, supervised_loss, net)),
    ), test, "train on one part, scored on another every few steps; Train stops when the score stops improving; report the third part")


def rag(k: int = 4) -> Script:
    """The same thing in plain Python:

        def rag(lm: Predictor, embed: Predictor, documents: Set, question: Temporal, *, k: int = 4) -> Temporal:
            vectors: Value = embed(documents)
            memory: Index = NewIndex(kind="ivf")(vectors)
            TrainIndex()(memory, vectors)
            Insert()(memory, Batches(size=1024, shuffle=False)(vectors))
            prompt: Temporal = Retrieve(k=k)(memory, documents, question, embed)
            return Walk(steps=64)(LMTree()(lm, prompt), Greedy()).end
    """
    lm, embed, documents, question, vectors, memory = inputs("lm embed documents question vectors memory")
    return Script("Retrieval-augmented generation", (lm, embed, documents, question), (
        let(vectors, embed(documents)),
        let(memory, call(NewIndex(kind="ivf"), vectors)),
        update(memory, TrainIndex(), vectors),
        update(memory, Insert(), call(Batches(size=1024, shuffle=False), vectors)),
    ), call(Walk(steps=64), call(LMTree(), lm, call(Retrieve(k=k), memory, documents, question, embed)), Greedy()).end,
        "CREATE INDEX on the documents' embeddings, then put the nearest ones in front of the question and decode")


PROGRAMS: list[Program | Script] = [
    language_model(), vit(), clip(), stable_diffusion(), mamba(), mixtral(), lora_finetune(), dpo(), alphazero(), r1(), diffusion_lm(),
    pretrain_finetune(), ppo(), dqn(), world_model_planning(), synthetic_data(), curation(), early_stopping(), rag(),
]

if __name__ == "__main__":
    for p in PROGRAMS:
        print(p, end="\n\n")
