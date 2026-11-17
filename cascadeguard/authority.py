"""Distinct effective rights, fixed normalization, and explicit grant provenance."""
from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction as Q
from typing import Iterable


def rational(value) -> Q:
    return Q(str(value)) if isinstance(value, float) else Q(value)


@dataclass(frozen=True, order=True)
class Permission:
    subject: str
    operation: str
    target: str
    category: str


@dataclass(frozen=True)
class AuthoritySpace:
    categories: tuple[str, ...]
    scales: tuple[Q, ...]
    baseline: frozenset[Permission] = frozenset()

    def __post_init__(self):
        object.__setattr__(self, "scales", tuple(map(rational, self.scales)))
        if (not self.categories or len(set(self.categories)) != len(self.categories)
                or len(self.scales) != len(self.categories) or min(self.scales) <= 0):
            raise ValueError("categories must be unique and scales strictly positive")
        self.vector(self.baseline)

    def vector(self, permissions: Iterable[Permission], *, additional=True) -> tuple[Q, ...]:
        rights = frozenset(permissions)
        if any(p.category not in self.categories for p in rights):
            raise ValueError("unknown permission category")
        if len({(p.subject, p.operation, p.target) for p in rights}) != len(rights):
            raise ValueError("one permission atom cannot belong to multiple categories")
        if additional:
            rights -= self.baseline
        return tuple(Q(sum(p.category == c for p in rights)) / scale
                     for c, scale in zip(self.categories, self.scales))


@dataclass
class Support:
    permission: Permission
    expires: Q
    premises: tuple[str, ...] = ()
    root: str | None = None
    present: bool = False
    absent: bool = False
    revoked: bool = False
    uid: str | None = None


class Ledger:
    """Single-caller model. Backend confirmations are trusted adapter inputs.

    AND prerequisites form an acyclic provenance graph; multiple supports for
    one permission are OR alternatives. Pending/expired grants remain counted
    until absence AND resolution of their writes have both been confirmed.
    """

    def __init__(self, baseline=(), max_ttl=60):
        self.baseline = frozenset(baseline)
        self.max_ttl = rational(max_ttl)
        if self.max_ttl <= 0:
            raise ValueError("positive root lifetime required")
        self.records: dict[str, Support] = {}
        self.roots: set[str] = set()
        self.clock = Q(0)

    def _time(self, now):
        now = rational(now)
        if now < self.clock:
            raise ValueError("time must be monotonic")
        self.clock = now
        return now

    def root(self, key, permission, *, now, expires):
        now, expires = self._time(now), rational(expires)
        if key in self.roots or key in self.records or not 0 < expires-now <= self.max_ttl:
            raise ValueError("root IDs are unique; renewal requires a new ID")
        self.roots.add(key)
        self.records[key] = Support(permission, expires, root=key)

    def valid(self, key, now):
        now = self._time(now)
        s = self.records[key]
        return (s.present and not s.absent and not s.revoked and now < s.expires
                and all(self.valid(p, now) for p in s.premises))

    def derive(self, key, permission, premises, *, now):
        now = self._time(now)
        premises = tuple(dict.fromkeys(premises))
        if key in self.records or not premises or not all(self.valid(p, now) for p in premises):
            raise PermissionError("new grant needs every existing, valid AND premise")
        self.records[key] = Support(permission, min(self.records[p].expires for p in premises),
                                    premises=premises)

    def confirm(self, key, uid, *, now):
        self._time(now)
        s = self.records[key]
        if (not uid or s.absent or s.uid not in (None, uid)
                or any(k != key and t.uid == uid for k, t in self.records.items())):
            raise ValueError("confirmation needs a unique, unchanged backend UID")
        s.present, s.uid = True, uid

    def revoke(self, key, *, now):
        self._time(now)
        self.records[key].revoked = True

    def confirm_absent(self, key, *, now, uid, write_resolved):
        now = self._time(now)
        s = self.records[key]
        if (not write_resolved or uid != s.uid
                or (not s.revoked and now < s.expires and all(self.valid(p, now) for p in s.premises))):
            raise ValueError("absence needs an invalidated grant and a resolved write")
        s.absent = True

    def effective(self):
        return self.baseline | frozenset(s.permission for s in self.records.values() if not s.absent)

    def authorized(self, permission, *, now):
        self._time(now)
        return permission in self.baseline or any(s.permission == permission and self.valid(k, now)
                                                 for k, s in self.records.items())
