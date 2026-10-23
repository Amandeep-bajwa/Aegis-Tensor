"""Static analysis of Jinja chat templates (GGUF ``tokenizer.chat_template*``).

A loader that renders a template in a plain (unsandboxed) Jinja environment runs any
Python the template can reach (CVE-2024-34359, llama-cpp-python before 0.2.72). Every
known escape walks Python internals through an attribute or item whose name starts
with ``_`` (``__class__``, ``__globals__``, ``_TemplateReference__context``), or
through ``str.format`` field access.

Matching text is not enough: the rules are public, so an attacker can build the name
at render time (``'_' * 2 ~ 'class' ~ '_' * 2``), hide it in a variable, a namespace
or a macro argument, or reach it through ``attr``/``map(attribute=...)``. This module
parses the template with Jinja's own parser and checks every place a name is looked
up. It never renders the template.

The rule: every attribute or item name must be visibly harmless. It must be a literal
that does not start with ``_``, or a value that comes from the user's render-time data
(messages, tools). A name that traces back to an underscore literal, that is built by a
string operation over template literals, or that comes from a model-supplied variable
(``bos_token``, ``eos_token``: their text is in the same file) is reported.
"""

from __future__ import annotations

import re
from typing import Dict, Iterator, List, Optional, Set, Tuple

# Filters that build or transform strings: a lookup key produced by one is computed.
_STRING_FILTERS = {
    "format", "join", "replace", "reverse", "string", "lower", "upper", "capitalize", "title", "trim",
    "center", "indent", "wordwrap", "truncate", "urlencode", "first", "last", "list", "safe", "e",
    "escape", "forceescape", "striptags", "batch", "slice", "sort", "unique", "map", "select", "reject",
}
# str methods that build strings.
_STRING_METHODS = {
    "join", "format", "format_map", "replace", "strip", "lstrip", "rstrip", "upper", "lower", "center",
    "ljust", "rjust", "zfill", "swapcase", "translate", "encode", "decode", "removeprefix", "removesuffix",
    "title", "capitalize", "casefold", "expandtabs", "partition", "rpartition", "split", "rsplit",
}
# Filters whose arguments name attributes or other filters.
_ATTRIBUTE_FILTERS = {"attr", "map", "select", "reject", "selectattr", "rejectattr", "sort", "groupby",
                      "sum", "unique", "min", "max", "dictsort"}
# Attributes that expose interpreter internals without a leading underscore.
_INTERNAL_ATTRIBUTES = {"mro", "func_globals", "func_code", "f_globals", "f_locals", "f_builtins", "gi_frame",
                        "gi_code", "cr_frame", "cr_code", "ag_frame", "ag_code", "tb_frame", "tb_next"}
_FORMAT_FIELD_ACCESS = re.compile(r"\{[^{}]*[.\[][^{}]*\}")

Issue = Tuple[str, str]  # (code, message): code "injection" or "unverifiable"


def check_template(text: str) -> List[Issue]:
    """Issues found in ``text``. An empty list means no escape route was found."""
    try:
        from jinja2 import Environment, nodes
        from jinja2.exceptions import TemplateSyntaxError
    except ImportError:  # pragma: no cover - jinja2 is a dependency; kept for minimal installs
        return _regex_fallback(text)
    try:
        tree = Environment(extensions=["jinja2.ext.loopcontrols", "jinja2.ext.do"]).parse(text)
    except TemplateSyntaxError as error:
        return [("unverifiable", f"chat template does not parse as Jinja ({error.message}, line {error.lineno})")]
    return _Analyzer(nodes, tree).run()


class _Analyzer:
    def __init__(self, nodes, tree):
        self.nodes, self.tree = nodes, tree
        self.issues: List[Issue] = []
        self.assignments: Dict[str, List] = {}  # variable or "ns.attr" -> expressions assigned to it
        self._collect_assignments()

    # --- Bindings -------------------------------------------------------------------

    def _bind(self, target, value) -> None:
        n = self.nodes
        if isinstance(target, n.Name):
            self.assignments.setdefault(target.name, []).append(value)
        elif isinstance(target, n.NSRef):
            self.assignments.setdefault(f"{target.name}.{target.attr}", []).append(value)
        elif isinstance(target, n.Tuple):
            for item in target.items:  # tuple unpacking: every element may come from anywhere in value
                self._bind(item, value)

    def _collect_assignments(self) -> None:
        n = self.nodes
        macros = {}
        for node in self.tree.find_all((n.Assign, n.AssignBlock, n.For, n.With, n.Macro, n.CallBlock)):
            if isinstance(node, n.Assign):
                self._bind(node.target, node.node)
                if isinstance(node.target, n.Name) and _is_call_to(n, node.node, "namespace"):
                    for keyword in node.node.kwargs:
                        self.assignments.setdefault(f"{node.target.name}.{keyword.key}", []).append(keyword.value)
            elif isinstance(node, n.AssignBlock):
                self._bind(node.target, n.Output(node.body))
            elif isinstance(node, n.For):
                self._bind(node.target, node.iter)
            elif isinstance(node, n.With):
                for target, value in zip(node.targets, node.values):
                    self._bind(target, value)
            elif isinstance(node, n.Macro):
                macros[node.name] = node
                offset = len(node.args) - len(node.defaults)
                for index, default in enumerate(node.defaults):
                    self._bind(node.args[offset + index], default)
            elif isinstance(node, n.CallBlock):
                for arg, default in zip(reversed(node.args), reversed(node.defaults)):
                    self._bind(arg, default)
        for call in self.tree.find_all(n.Call):  # macro arguments are bound at each call site
            if isinstance(call.node, n.Name) and call.node.name in macros:
                params = macros[call.node.name].args
                for param, value in zip(params, call.args):
                    self._bind(param, value)
                for keyword in call.kwargs:
                    for param in params:
                        if param.name == keyword.key:
                            self._bind(param, keyword.value)

    # --- Taint ----------------------------------------------------------------------

    def _resolve(self, node, seen: Set[str]) -> Iterator:
        """Expressions assigned to the variable (or namespace attribute) ``node`` reads."""
        n = self.nodes
        key = None
        if isinstance(node, n.Name) and node.ctx == "load":
            key = node.name
        elif isinstance(node, n.Getattr) and isinstance(node.node, n.Name):
            key = f"{node.node.name}.{node.attr}"
        if key and key in self.assignments and key not in seen:
            seen.add(key)
            for value in self.assignments[key]:
                yield key, value

    def _walk_resolved(self, expr, seen: Set[str]) -> Iterator:
        for node in _walk(self.nodes, expr):
            yield node
            for _, value in self._resolve(node, seen):
                yield from self._walk_resolved(value, seen)

    def _has_literal(self, expr) -> bool:
        n = self.nodes
        return any((isinstance(node, n.Const) and isinstance(node.value, str))
                   or _model_supplied(n, node, self.assignments)
                   for node in self._walk_resolved(expr, set()))

    def _taint(self, expr) -> Optional[str]:
        """Why ``expr`` is not a visibly harmless name, or None if it is.

        A name is harmful if it traces back to an underscore literal, or if it is built by a
        string operation from literals in the template. Names built only from render-time
        data (the user's messages and tools) cannot be chosen by the template author.
        """
        n = self.nodes
        for node in self._walk_resolved(expr, set()):
            if isinstance(node, n.Const) and isinstance(node.value, str) and _underscore(node.value):
                return f"the literal {node.value!r}"
            if _model_supplied(n, node, self.assignments):
                return f"`{node.name}`, whose text comes from the model file"
            built = None
            if isinstance(node, n.Concat):
                built = "string concatenation"
            elif isinstance(node, (n.Add, n.Mul, n.Mod)):
                built = "string arithmetic"
            elif isinstance(node, n.Filter) and node.name in _STRING_FILTERS:
                built = f"the `{node.name}` filter"
            elif isinstance(node, n.Call) and isinstance(node.node, n.Getattr) and node.node.attr in _STRING_METHODS:
                built = f"`.{node.node.attr}()`"
            if built and self._has_literal(node):
                return f"template literals through {built}"
        return None

    # --- Checks ---------------------------------------------------------------------

    def _report(self, code: str, message: str, node) -> None:
        line = getattr(node, "lineno", None)
        self.issues.append((code, message + (f" (line {line})" if line else "")))

    def _check_name(self, expr, where: str, node) -> None:
        n = self.nodes
        if isinstance(expr, n.Const):
            if isinstance(expr.value, str) and any(_underscore(part) for part in expr.value.split(".")):
                self._report("injection", f"{where} reaches the internal name {expr.value!r}", node)
            return
        reason = self._taint(expr)
        if reason:
            self._report("injection", f"{where} is computed at render time from {reason}", node)

    def run(self) -> List[Issue]:
        n = self.nodes
        for node in self.tree.find_all((n.Import, n.FromImport, n.Include, n.Extends)):
            self._report("unverifiable", f"template loads another template (`{type(node).__name__.lower()}`)", node)
        for node in self.tree.find_all(n.Getattr):
            if _underscore(node.attr) or node.attr in _INTERNAL_ATTRIBUTES:
                self._report("injection", f"attribute `.{node.attr}` reaches Python internals", node)
        for node in self.tree.find_all(n.Getitem):
            if not isinstance(node.arg, n.Slice):
                self._check_name(node.arg, "an item lookup", node)
        for node in self.tree.find_all(n.Filter):
            if node.name in _ATTRIBUTE_FILTERS:
                if node.name in ("map", "select", "reject") and node.args and _const(n, node.args[0]) == "attr":
                    self._report("injection", f"`{node.name}('attr', ...)` looks attributes up by name", node)
                for arg in [*node.args, *(k.value for k in node.kwargs)]:
                    self._check_name(arg, f"the `{node.name}` filter's argument", node)
        for node in self.tree.find_all(n.Call):
            if isinstance(node.node, n.Getattr) and node.node.attr in ("format", "format_map"):
                fmt = node.node.node
                if not isinstance(fmt, n.Const):
                    self._report("injection", "`str.format` is called on a computed format string", node)
                elif isinstance(fmt.value, str) and _FORMAT_FIELD_ACCESS.search(fmt.value):
                    self._report("injection", f"`str.format` field {fmt.value!r} reads attributes of its arguments", node)
        seen, unique = set(), []
        for issue in self.issues:
            if issue not in seen:
                seen.add(issue)
                unique.append(issue)
        return unique


def _walk(nodes, expr) -> Iterator:
    yield expr
    for child in expr.iter_child_nodes():
        yield from _walk(nodes, child)


def _model_supplied(nodes, node, assignments) -> bool:
    """Render variables such as ``bos_token`` and ``eos_token`` are read from the model's own
    vocabulary, so the file's author chooses their text."""
    return (isinstance(node, nodes.Name) and node.ctx == "load" and node.name.endswith("_token")
            and node.name not in assignments)


def _underscore(name: str) -> bool:
    return name.startswith("_")


def _const(nodes, expr):
    return expr.value if isinstance(expr, nodes.Const) else None


def _is_call_to(nodes, expr, name: str) -> bool:
    return isinstance(expr, nodes.Call) and isinstance(expr.node, nodes.Name) and expr.node.name == name


_REGEX_FALLBACK = [
    (re.compile(r"__[A-Za-z]+__"), "dunder attribute access"),
    (re.compile(r"\|\s*attr\s*\("), "`attr` filter"),
]


def _regex_fallback(text: str) -> List[Issue]:
    hits = [label for pattern, label in _REGEX_FALLBACK if pattern.search(text)]
    issues = [("injection", "chat template uses " + ", ".join(hits))] if hits else []
    return issues + [("unverifiable", "jinja2 is not installed, so only a text match was possible")]
