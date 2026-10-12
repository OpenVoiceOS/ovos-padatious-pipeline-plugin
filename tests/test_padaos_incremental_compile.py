"""A padaos compile rebuilds only what changed since the previous pass.

Skills register one after another, and every registration dirties the
container, so a full rebuild per pass made boot quadratic in the number of
skills (once per language). These tests pin that an unchanged intent keeps
the exact regex objects of the previous pass, and that everything that can
change an intent's regexes - its own lines, an entity it references, an entity
crossing the inline cap - still rebuilds it.
"""
import unittest
from unittest import mock

from ovos_padatious import padaos


def _matches(container, query):
    return sorted(m["name"] for m in container.calc_intents(query))


class TestIncrementalCompile(unittest.TestCase):

    def test_unchanged_intent_is_not_rebuilt(self):
        c = padaos.IntentContainer()
        c.add_intent("skill_a:greet", ["hello there", "hi (there|friend)"])
        c.compile()
        first = c.intents["skill_a:greet"]

        c.add_intent("skill_b:bye", ["goodbye now"])
        with mock.patch.object(padaos.IntentContainer, "create_regexes",
                               wraps=c.create_regexes) as spy:
            c.compile()
        built = [call.args[1] for call in spy.call_args_list]
        self.assertEqual(built, ["skill_b:bye"])
        self.assertIs(c.intents["skill_a:greet"], first)
        self.assertEqual(_matches(c, "hi friend"), ["skill_a:greet"])
        self.assertEqual(_matches(c, "goodbye now"), ["skill_b:bye"])

    def test_changed_lines_rebuild_the_intent(self):
        c = padaos.IntentContainer()
        c.add_intent("greet", ["hello there"])
        c.compile()
        c.add_intent("greet", ["good morning"])
        c.compile()
        self.assertEqual(_matches(c, "good morning"), ["greet"])
        self.assertEqual(_matches(c, "hello there"), [])

    def test_referenced_entity_change_rebuilds_only_its_intents(self):
        c = padaos.IntentContainer()
        c.add_intent("paint", ["paint it {colour}"])
        c.add_intent("greet", ["hello there"])
        c.add_entity("colour", ["red"])
        c.compile()
        greet = c.intents["greet"]
        self.assertEqual(c.calc_intent("paint it red")["entities"], {"colour": "red"})

        c.add_entity("colour", ["blue"])
        c.compile()
        self.assertIs(c.intents["greet"], greet)
        self.assertEqual(c.calc_intent("paint it blue")["entities"], {"colour": "blue"})
        self.assertEqual(_matches(c, "paint it red"), [])

    def test_namespaced_entity_is_tracked(self):
        c = padaos.IntentContainer()
        c.add_intent("skill:paint", ["paint it {colour}"])
        c.add_entity("skill:colour", ["red"])
        c.compile()
        self.assertEqual(_matches(c, "paint it red"), ["skill:paint"])
        c.add_entity("skill:colour", ["green"])
        c.compile()
        self.assertEqual(_matches(c, "paint it green"), ["skill:paint"])
        self.assertEqual(_matches(c, "paint it red"), [])

    def test_entity_registered_later_replaces_the_wildcard(self):
        c = padaos.IntentContainer()
        c.add_intent("paint", ["paint it {colour}"])
        c.compile()
        # no entity yet: the slot is a wildcard
        self.assertEqual(_matches(c, "paint it purple"), ["paint"])
        c.add_entity("colour", ["red"])
        c.compile()
        self.assertEqual(_matches(c, "paint it purple"), [])
        self.assertEqual(_matches(c, "paint it red"), ["paint"])

    def test_entity_crossing_the_inline_cap_rebuilds_its_intents(self):
        c = padaos.IntentContainer()
        c.add_intent("paint", ["paint it {colour}"])
        c.add_entity("colour", ["red"])
        c.compile()
        self.assertEqual(_matches(c, "paint it purple"), [])
        values = ["c%d" % i for i in range(padaos.PADAOS_ENTITY_INLINE_CAP + 1)]
        c.add_entity("colour", values)
        c.compile()
        self.assertIn("colour", c.capped_entities)
        # capped: the slot falls back to the wildcard capture
        self.assertEqual(_matches(c, "paint it purple"), ["paint"])

    def test_removed_names_leave_the_memo(self):
        c = padaos.IntentContainer()
        c.add_intent("greet", ["hello there"])
        c.add_entity("colour", ["red"])
        c.compile()
        c.remove_intent("greet")
        c.remove_entity("colour")
        c.compile()
        self.assertNotIn("greet", c._intent_cache)
        self.assertNotIn("colour", c._entity_cache)
        self.assertEqual(_matches(c, "hello there"), [])

    def test_group_names_unique_within_each_intent(self):
        c = padaos.IntentContainer()
        c.add_intent("move", ["move {a} to {b}", "put {a} on {b}"])
        c.add_intent("copy", ["copy {a} to {b}"])
        c.compile()
        self.assertEqual(c.calc_intent("move box to shelf")["entities"],
                         {"a": "box", "b": "shelf"})
        self.assertEqual(c.calc_intent("copy x to y")["entities"],
                         {"a": "x", "b": "y"})


if __name__ == "__main__":
    unittest.main()


class TestEntityOrderInTheKey(unittest.TestCase):
    def test_reordered_colliding_entities_rebuild_the_intent(self):
        """Both 'colour' and 'skill:colour' fill {colour} in skill:paint; the
        one later in the entities' order wins, so re-adding the first one
        (which moves it last) must rebuild the intent."""
        c = padaos.IntentContainer()
        c.add_intent("skill:paint", ["paint it {colour}"])
        c.add_entity("colour", ["red"])
        c.add_entity("skill:colour", ["blue"])
        c.compile()
        fresh = padaos.IntentContainer()
        fresh.add_intent("skill:paint", ["paint it {colour}"])
        fresh.add_entity("skill:colour", ["blue"])
        fresh.add_entity("colour", ["red"])
        fresh.compile()

        c.remove_entity("colour")
        c.add_entity("colour", ["red"])  # now after skill:colour, as in `fresh`
        c.compile()
        for query in ("paint it red", "paint it blue"):
            self.assertEqual(_matches(c, query), _matches(fresh, query), query)
