"""Independent finite-effect enumeration and safety regression tests."""
from dataclasses import replace
from fractions import Fraction as Q
from itertools import combinations
import unittest

from cascadeguard.authority import AuthoritySpace, Ledger, Permission
from cascadeguard.contracts import Contract, Rule, compile_contracts
from cascadeguard.certification import Certificate, certify, optimize
from cascadeguard.runtime import EpochGate, Evaluation, Guard, Request, Stage, Trace
from demo import model, run_demo


class AuthorityTests(unittest.TestCase):
    def setUp(self):
        self.a = Permission("O", "create", "ns/a", "x")
        self.b = Permission("I", "bind", "ns/b", "y")
        self.c = Permission("D", "get", "other/c", "y")
        self.space = AuthoritySpace(("x", "y"), (Q(2), Q(3)))

    def test_deduplication_baseline_and_scales(self):
        space = replace(self.space, baseline=frozenset({self.a}))
        self.assertEqual(space.vector([self.a, self.a, self.b, self.b]), (0, Q(1, 3)))
        with self.assertRaises(ValueError):
            AuthoritySpace(("x",), (0,))

    def test_independent_all_subsets_cover_and_or_effects(self):
        rules = (Rule("and", "m", "grant", frozenset({self.a, self.b}), frozenset({self.c})),
                 Rule("or", "m", "grant", frozenset({self.b}), frozenset({self.c})))
        contract = compile_contracts(self.space, rules)[0]
        universe = (self.a, self.b, self.c)
        for size in range(len(universe)+1):
            for subset in combinations(universe, size):
                rights = frozenset(subset)
                # Direct Boolean effect evaluation; do not use AGC predictions as truth.
                output = frozenset().union(*(r.outputs for r in rules if r.premises <= rights))
                actual = self.space.vector(output)
                x = self.space.vector(rights)
                bound = tuple(sum(a*v for a, v in zip(row, x)) for row in contract.matrix)
                self.assertTrue(all(a <= b for a, b in zip(actual, bound)))

    def test_all_modes_need_one_common_witness(self):
        rules = (Rule("xy", "forward", "grant", frozenset({self.a}), frozenset({self.b})),
                 Rule("yx", "return", "grant", frozenset({self.b}), frozenset({self.a})))
        contracts = compile_contracts(self.space, rules)
        self.assertTrue(all(certify((c,), 2) is not None for c in contracts))
        self.assertIsNone(certify(contracts, 2))
        self.assertFalse(Certificate((Q(1, 2), Q(1, 2)), Q(4, 5), Q(1, 100)).verifies(contracts))

    def test_cumulative_limit_is_max_not_addition(self):
        contract = Contract("m", ((Q(1), Q(0)), (Q(0), Q(0))), frozenset({0}))
        bound = contract.bound((1, 0), (1, 0))
        self.assertEqual(bound, (1, 0))
        self.assertGreater(2, bound[0])  # Different new right cannot fit beside retained right.
        with self.assertRaises(ValueError):
            contract.bound((1, 0), (2, 0))

    def test_optimizer_generates_subset_and_preserves_task_minima(self):
        space, _, rules, required = model()
        result = optimize(space, rules, required=required)
        self.assertIsNotNone(result)
        self.assertTrue(required <= result.enabled < frozenset(r.key for r in rules))
        self.assertEqual(result.retained, Q(4, 5))
        self.assertEqual(result.candidates_checked, 4)
        self.assertTrue(result.certificate.verifies(result.contracts))

    def test_safe_full_configuration_is_not_unnecessarily_restricted(self):
        rule = Rule("xy", "m", "grant", frozenset({self.a}), frozenset({self.b}))
        result = optimize(self.space, (rule,), required=frozenset({"xy"}))
        self.assertEqual(result.enabled, frozenset({"xy"}))
        self.assertEqual(result.retained, 1)


class RuntimeTests(unittest.TestCase):
    def setUp(self):
        self.a = Permission("I", "read", "ns/a", "read")
        self.b = Permission("I", "read", "ns/b", "read")
        self.space = AuthoritySpace(("read",), (Q(1),))
        self.guard = Guard(self.space, ((1,),), (1,))
        self.contract = Contract("m", ((Q(0),),), frozenset({0}))
        self.original = Request("O", ("grant", "revoke"), frozenset({"a", "b"}))

    def evaluation(self, stages, **kwargs):
        return Evaluation(True, True, (Trace(self.contract, tuple(stages)),), **kwargs)

    def test_all_intermediate_states_checked(self):
        trace = self.evaluation((Stage(frozenset({self.a, self.b}), root_supply=(Q(2),)),
                                 Stage(frozenset(), root_supply=(Q(0),))))
        self.assertFalse(self.guard.accepts(frozenset(), trace))

    def test_new_intermediate_permission_also_needs_confirmed_revocation(self):
        missing_ack = self.evaluation((Stage(frozenset({self.a}), root_supply=(Q(1),)),
                                        Stage(frozenset(), root_supply=(Q(0),))))
        self.assertFalse(self.guard.accepts(frozenset(), missing_ack))
        resolved = self.evaluation((Stage(frozenset({self.a}), root_supply=(Q(1),)),
                                   Stage(frozenset(), frozenset({self.a}), (Q(0),))))
        self.assertTrue(self.guard.accepts(frozenset(), resolved))

    def test_boundary_counts_unnormalized_baseline_permissions(self):
        space = AuthoritySpace(("read",), (Q(2),), frozenset({self.a, self.b}))
        guard = Guard(space, ((1,),), (2,))
        self.assertTrue(guard.safe((Q(0),)))
        self.assertFalse(guard.safe((Q(1, 2),)))

    def test_pending_permission_cannot_disappear_without_confirmation(self):
        result = self.evaluation((Stage(frozenset()),))
        self.assertFalse(self.guard.accepts(frozenset({self.a}), result))
        confirmed = self.evaluation((Stage(frozenset(), confirmed_absent=frozenset({self.a})),))
        self.assertTrue(self.guard.accepts(frozenset({self.a}), confirmed))

    def test_unknown_coverage_and_unsafe_alternative_fail_closed(self):
        safe = Trace(self.contract, (Stage(frozenset()),))
        unsafe = Trace(self.contract, (Stage(frozenset({self.a, self.b}), root_supply=(Q(2),)),))
        self.assertFalse(self.guard.accepts((), Evaluation(True, False, (safe,))))
        self.assertFalse(self.guard.accepts((), Evaluation(True, True, (safe, unsafe))))

    def test_order_reversal_is_not_narrowing(self):
        reversed_order = replace(self.original, operations=("revoke", "grant"))
        with self.assertRaises(ValueError):
            self.guard.choose(self.original, (self.original, reversed_order), (),
                              lambda r: self.evaluation((Stage(frozenset()),)))

    def test_original_scope_wins_when_safe(self):
        smaller = replace(self.original, targets=frozenset({"a"}))
        chosen = self.guard.choose(self.original, (smaller, self.original), (),
                                   lambda r: self.evaluation((Stage(frozenset()),)))
        self.assertEqual(chosen, self.original)


class ProvenanceTests(unittest.TestCase):
    def setUp(self):
        self.a = Permission("O", "create", "ns/a", "x")
        self.b = Permission("O", "bind", "ns/a", "x")
        self.out = Permission("I", "get", "ns/a", "y")
        self.ledger = Ledger(max_ttl=10)
        for key, permission, expiry in (("a", self.a, 10), ("b", self.b, 8)):
            self.ledger.root(key, permission, now=0, expires=expiry)
            self.ledger.confirm(key, "uid-"+key, now=0)

    def test_cross_role_and_premises_expiry_and_pending_accounting(self):
        self.ledger.derive("out", self.out, ("a", "b"), now=1)
        self.assertIn(self.out, self.ledger.effective())
        self.assertFalse(self.ledger.authorized(self.out, now=1))
        self.ledger.confirm("out", "uid-out", now=1)
        self.assertTrue(self.ledger.authorized(self.out, now=1))
        self.assertFalse(self.ledger.authorized(self.out, now=8))
        self.assertIn(self.out, self.ledger.effective())
        with self.assertRaises(ValueError):
            self.ledger.confirm_absent("out", now=8, uid="uid-out", write_resolved=False)
        self.ledger.confirm_absent("out", now=8, uid="uid-out", write_resolved=True)
        self.assertNotIn(self.out, self.ledger.effective())

    def test_or_supports_deduplicate_and_root_ids_cannot_be_renewed(self):
        self.ledger.derive("one", self.out, ("a",), now=1)
        self.ledger.derive("two", self.out, ("b",), now=1)
        self.ledger.confirm("one", "uid-one", now=1)
        self.ledger.confirm("two", "uid-two", now=1)
        self.ledger.revoke("a", now=2)
        self.assertTrue(self.ledger.authorized(self.out, now=2))
        self.assertEqual(sum(p == self.out for p in self.ledger.effective()), 1)
        with self.assertRaises(ValueError):
            self.ledger.root("a", self.a, now=2, expires=9)


class DemoTests(unittest.TestCase):
    def test_five_identity_execution_and_real_scope_subset(self):
        result = run_demo()
        self.assertFalse(result["q0_certified"])
        self.assertTrue(result["certificate_exactly_verified"])
        self.assertTrue(result["task_completed"])
        self.assertEqual([d["actor"] for d in result["decisions"]], list("OIDNR"))
        self.assertTrue(result["decisions"][0]["narrowed"])

    def test_epoch_recertification_and_stale_requests(self):
        space, _, rules, required = model()
        configuration = optimize(space, rules, required=required)
        gate = EpochGate(configuration, Guard(space, (), ()))
        snapshot = lambda: (frozenset(), b"complete-provenance-and-pending-snapshot")
        self.assertEqual(gate.activate(0, snapshot=snapshot, compile_configuration=lambda s: configuration), 1)
        with self.assertRaises(RuntimeError):
            gate.call(0, lambda c: True)
        with self.assertRaises(ValueError):
            gate.activate(1, snapshot=snapshot, compile_configuration=lambda s: None)
        self.assertEqual(gate.version, 1)


if __name__ == "__main__":
    unittest.main()
