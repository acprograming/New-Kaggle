"""Independent CPU checks of pure cache-contract and prospective-split helpers.

Extract the actual pure functions so these checks run without importing Torch.
The MRI Dataset and model still require separate real Torch/GPU qualification.
"""
import ast
import hashlib
from pathlib import Path
import re
import unittest

import numpy as np


def load_helpers():
    source = Path(__file__).resolve().parents[1] / "src" / "rsna" / "cache_data.py"
    tree = ast.parse(source.read_text())
    wanted = [node for node in tree.body if isinstance(node, ast.FunctionDef)
              and node.name in {"eligible_centers", "group_split"}]
    assert len(wanted) == 2
    env = {"np": np, "hashlib": hashlib, "re": re, "SLOT_WIDTHS": (12, 10, 8, 6, 8)}
    exec(compile(ast.Module(body=wanted, type_ignores=[]), str(source), "exec"), env)
    return env["eligible_centers"], env["group_split"]


class CacheSplitContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.centers, cls.split = map(staticmethod, load_helpers())

    def test_triplets_cover_every_allowed_center_without_crossing_slots(self):
        rows = self.centers(np.ones(44, np.uint8))
        self.assertEqual(rows.shape, (34, 3))
        bounds = np.r_[0, np.cumsum((12, 10, 8, 6, 8))]
        for center, slot, position in rows:
            left, right = bounds[int(slot):int(slot)+2]
            self.assertGreaterEqual(center-1, left)
            self.assertLess(center+1, right)
            self.assertGreater(position, -1)
            self.assertLess(position, 1)
        broken = np.ones(44, np.uint8)
        broken[5] = 0
        rows = self.centers(broken)
        self.assertTrue(all(5 not in range(int(c)-1, int(c)+2) for c in rows[:, 0]))

    def test_report_and_known_relation_components_cannot_escape_gold_exclusion(self):
        ids = np.array([f"synthetic-{i}" for i in range(100)])
        reports = [f"unique synthetic report {i}" for i in range(100)]
        reports[1], reports[2] = "same report", " SAME   REPORT "
        reports[4] = ""
        gold = np.zeros(100, bool)
        gold[1] = True
        groups, roles, folds = self.split(ids, reports, gold, seed=123, n_folds=4,
                                          holdout_fraction=.2, relations=[(ids[2], ids[3])])
        self.assertEqual(len(set(groups[1:4])), 1)
        self.assertTrue((roles[1:4] == "excluded").all())
        self.assertEqual(roles[4], "excluded")
        self.assertTrue((folds[roles != "development"] == -1).all())
        for group in np.unique(groups):
            selected = groups == group
            self.assertEqual(len(set(roles[selected])), 1)
            self.assertEqual(len(set(folds[selected])), 1)

    def test_locked_hash_assignment_does_not_depend_on_input_row_order(self):
        ids = np.array([f"synthetic-{i}" for i in range(100)])
        reports = np.array([f"report {i//2}" for i in range(100)])
        gold = np.zeros(100, bool)
        forward = self.split(ids, reports, gold, seed=456, n_folds=4, holdout_fraction=.2)
        order = np.random.default_rng(87).permutation(100)
        shuffled = self.split(ids[order], reports[order], gold[order], seed=456, n_folds=4, holdout_fraction=.2)
        undo = np.argsort(order)
        for a, b in zip(forward, shuffled):
            np.testing.assert_array_equal(a, b[undo])

    def test_unknown_relations_and_exclusions_are_rejected(self):
        ids, reports, gold = ["one", "two"], ["a", "b"], [False, False]
        common = {"seed": 1, "n_folds": 4, "holdout_fraction": .2}
        with self.assertRaises(ValueError):
            self.split(ids, reports, gold, relations=[("one", "unknown")], **common)
        with self.assertRaises(ValueError):
            self.split(ids, reports, gold, excluded_ids=["unknown"], **common)


if __name__ == "__main__":
    unittest.main()
