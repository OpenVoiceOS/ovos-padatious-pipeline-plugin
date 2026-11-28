"""Containers sharing one pattern cache file keep each other's lines.

padatious' own pipeline, ovos-ocp-pipeline-plugin and ovos-persona each make
an ``IntentContainer`` on the same ``intent_cache/<lang>/`` folder, so their
padaos containers share ``padaos-patterns.json``. Each one used to write
only the lines it compiled, so whichever saved last threw the others' lines
away, and the next start rewrote them all.
"""
import os
import tempfile
import unittest

from ovos_padatious import padaos


def compiled(path, name, lines):
    c = padaos.IntentContainer(pattern_cache=path)
    c.add_intent(name, lines)
    c.compile()
    return c


class TestSharedPatternCache(unittest.TestCase):

    def test_a_later_container_keeps_the_earlier_ones_lines(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "padaos-patterns.json")
            compiled(path, "big:time", ["what time is it in {place}", "time in {place} please"])
            compiled(path, "small:play", ["start {query}"])

            fresh = padaos._PatternCache(path)
            known = fresh._load()
            for line in ("what time is it in {place}", "time in {place} please", "start {query}"):
                self.assertIn(line, known)

    def test_kept_lines_are_used_without_being_rewritten(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "padaos-patterns.json")
            compiled(path, "big:time", ["what time is it in {place}"])
            compiled(path, "small:play", ["start {query}"])

            c = padaos.IntentContainer(pattern_cache=path)
            calls = []
            original = c._create_pattern
            c._create_pattern = lambda line: calls.append(line) or original(line)
            c.add_intent("big:time", ["what time is it in {place}"])
            c.compile()
            self.assertEqual(calls, [])
            self.assertEqual(c.calc_intent("what time is it in lisbon")["entities"],
                             {"place": "lisbon"})
