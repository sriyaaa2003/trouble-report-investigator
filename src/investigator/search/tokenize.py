"""Tokenisation tuned for engineering text: code identifiers, stack frames, error strings.

`mozilla::dom::ContentParent::RecvFoo` yields the full identifier AND its parts
(`mozilla`, `dom`, `contentparent`, `content`, `parent`, `recvfoo`, ...), so exact symbol matches score
highly while partial matches still help.
"""

from __future__ import annotations

import re

_IDENT = re.compile(r"[A-Za-z_][A-Za-z0-9_]*(?:(?:::|\.)[A-Za-z_][A-Za-z0-9_]*)*")
_CAMEL = re.compile(r"[A-Z]+(?=[A-Z][a-z])|[A-Z]?[a-z]+|[A-Z]+|\d+")

_STOP = frozenset(
    (
        "a an the and or but if then else of to in on at by for with from as is are was were be been being it its "
        "this that these those i you we they he she not no do does did have has had can could should would will "
        "just also so than there here when while which who what where how all any some more most other into out "
        "up down over under again"
    ).split()
)


def tokenize(text: str) -> list[str]:
    out: list[str] = []
    for m in _IDENT.finditer(text):
        ident = m.group(0)
        low = ident.lower()
        if low in _STOP or len(low) < 2:
            continue
        out.append(low)
        parts = re.split(r"::|\.", ident)
        if len(parts) > 1:  # also emit each namespace/class segment whole (e.g. "contentparent")
            out += [p.lower() for p in parts if len(p) > 1 and p.lower() not in _STOP and p.lower() != low]
        sub: list[str] = []
        for part in parts:
            for seg in part.split("_"):
                sub += [w.lower() for w in _CAMEL.findall(seg)]
        if len(sub) > 1 or (len(sub) == 1 and sub[0] != low):
            out += [s for s in sub if len(s) > 1 and s not in _STOP and not s.isdigit()]
    return out
