"""padaos builds the same regex for the same template line at every start.

The patterns used to depend on the order an intent's lines and an entity's
values arrived in (which follows the skill's expanded samples and is not
the same from one start to the next), on the order of the entities dict,
and on the iteration order of a set() of slot names (string hashing is
randomised per process). Nothing matched differently, but nothing that
keys on the pattern text - a cache of compiled regexes, say - could ever
recognise a line it had seen at the last start.
"""
import json
import os
import subprocess
import sys
import unittest

from ovos_padatious import padaos

VALUES = ["groceries", "todo", "to do", "shopping"]
LINES = ["delete the {list_name} list", "remove the {list_name} list from {where}",
         "clear {list_name} on {device} {when}", "empty my {list_name} list"]


def patterns(values, lines, other_first=False):
    c = padaos.IntentContainer()
    if other_first:
        c.add_entity("other.skill:colour", ["red", "green"])
    c.add_entity("list.skill:list_name", values)
    c.add_entity("list.skill:device", ["phone", "tablet", "kitchen speaker"])
    if not other_first:
        c.add_entity("other.skill:colour", ["red", "green"])
    c.add_intent("list.skill:delete", lines)
    c.compile()
    return [r.pattern for r in c.intents["list.skill:delete"]], c


class TestStablePatterns(unittest.TestCase):

    def test_order_of_values_lines_and_entities_does_not_change_the_patterns(self):
        a, _ = patterns(VALUES, LINES)
        b, _ = patterns(list(reversed(VALUES)), list(reversed(LINES)), other_first=True)
        self.assertEqual(a, b)

    def test_same_patterns_in_another_process(self):
        """set() order follows the per-process string hash seed."""
        code = ("import json, sys; sys.path.insert(0, %r); "
                "from tests.test_padaos_stable_patterns import patterns, VALUES, LINES; "
                "print(json.dumps(patterns(VALUES, LINES)[0]))") % os.getcwd()
        runs = set()
        for seed in ("1", "2", "3", "4"):
            out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                                 env={**os.environ, "PYTHONHASHSEED": seed}, check=True)
            runs.add(out.stdout.strip().splitlines()[-1])
        self.assertEqual(len(runs), 1, runs)

    def test_still_matches_the_same(self):
        _, c = patterns(VALUES, LINES)
        m = c.calc_intent("delete the to do list")
        self.assertEqual(m["name"], "list.skill:delete")
        self.assertEqual(m["entities"], {"list_name": "to do"})
        m = c.calc_intent("clear shopping on kitchen speaker tonight")
        self.assertEqual(m["entities"], {"list_name": "shopping", "device": "kitchen speaker",
                                         "when": "tonight"})
        m = c.calc_intent("remove the groceries list from the fridge")
        self.assertEqual(m["entities"], {"list_name": "groceries", "where": "the fridge"})
        # the prefilter joins all lines of the intent: group names stay unique
        self.assertIsNotNone(c.intents["list.skill:delete"].prefilter)
