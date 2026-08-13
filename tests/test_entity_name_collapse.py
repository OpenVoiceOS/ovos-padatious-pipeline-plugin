"""Entity-name collapse on the legacy ``padatious:register_entity`` contract.

ovos-workshop's ``register_entity_file`` names the entity
``<skill_id>:<basename>_<md5(entity_file)>`` and puts that munged name on the
legacy ``padatious:register_entity`` topic verbatim. Slot lookup
(``EntityManager.find``) builds the candidate key from the intent's skill_id
plus the RAW slot token from the template, so it can never find the
hash-suffixed entry: every file-registered entity ended up as an
unconstrained wildcard slot.

This plugin owns its lookup contract, so it collapses the munged name at
REGISTRATION time — which fixes every emitter vintage, including the old
workshop releases that will keep emitting the munged name forever.
"""
import unittest

from ovos_bus_client.message import Message
from ovos_utils.messagebus import FakeBus

from ovos_padatious.opm import PadatiousPipeline

SKILL_ID = "entity.collapse.skill"
LANG = "en-US"
# the exact name ovos-workshop's register_entity_file() builds for
# "weekend.entity": md5("weekend") == 4ca4f434da0ea97ebff27833d69728d3
ENTITY_FILE = "weekend"
MUNGED_NAME = f"{SKILL_ID}:weekend_4ca4f434da0ea97ebff27833d69728d3"
CLEAN_NAME = f"{SKILL_ID}:weekend"
INTENT_NAME = f"{SKILL_ID}:ask.day"

ENTITY_SAMPLES = ["saturday", "sunday"]
INTENT_SAMPLES = ["is it {weekend}"]


def _wrapped(name):
    """Engine-side name: ``<skill_id>:{<entity>}`` (Entity.wrap_name)."""
    skill_id, ent = name.split(":", 1)
    return f"{skill_id}:{{{ent}}}"


def _md5_check():
    from hashlib import md5
    return md5(ENTITY_FILE.encode("utf-8")).hexdigest()


class TestLegacyEntityNameCollapse(unittest.TestCase):
    """Red -> green: the REAL munged legacy payload must constrain the slot."""

    def setUp(self):
        self.pipeline = PadatiousPipeline(
            FakeBus(), config={"instant_train": True})

    def tearDown(self):
        self.pipeline.shutdown()

    def test_fixture_matches_workshop_naming(self):
        # guards the fixture: if workshop's hash input ever changes, this
        # test tells you the payload under test is no longer the real one
        self.assertEqual(MUNGED_NAME,
                         f"{SKILL_ID}:{ENTITY_FILE}_{_md5_check()}")

    def _register(self):
        self.pipeline.register_entity(Message(
            "padatious:register_entity",
            {"skill_id": SKILL_ID, "name": MUNGED_NAME, "lang": LANG,
             "samples": ENTITY_SAMPLES},
            {"skill_id": SKILL_ID}))
        self.pipeline.register_intent(Message(
            "padatious:register_intent",
            {"skill_id": SKILL_ID, "name": INTENT_NAME, "lang": LANG,
             "samples": INTENT_SAMPLES},
            {"skill_id": SKILL_ID}))
        self.pipeline.train(Message("mycroft.skills.train"))

    def test_munged_legacy_entity_constrains_the_slot(self):
        self._register()
        container = self.pipeline.containers[LANG]

        good = container.calc_intent("is it saturday")
        self.assertEqual(good.name, INTENT_NAME)
        self.assertEqual(good.matches.get("weekend"), "saturday")

        # out-of-entity value must NOT fill the slot with high confidence
        bad = container.calc_intent("is it pizza")
        self.assertLess(bad.conf, good.conf)

    def test_entity_is_registered_under_the_clean_name(self):
        self._register()
        container = self.pipeline.containers[LANG]
        names = [e.name for e in container.entities.objects
                 + container.entities.objects_to_train]
        self.assertIn(_wrapped(CLEAN_NAME), names)
        self.assertNotIn(_wrapped(MUNGED_NAME), names)


class TestEntityRegistrationExactlyOnce(unittest.TestCase):
    """F3: workshop >= 9.3 dual-emits one entity (munged legacy twin + clean
    spec twin). Once both collapse to the same name, registration must stay
    idempotent — one manifest entry, one engine object.
    """

    def setUp(self):
        self.pipeline = PadatiousPipeline(FakeBus())

    def tearDown(self):
        self.pipeline.shutdown()

    def _dual_register(self):
        self.pipeline.register_entity(Message(
            "padatious:register_entity",
            {"skill_id": SKILL_ID, "name": MUNGED_NAME, "lang": LANG,
             "samples": ENTITY_SAMPLES},
            {"skill_id": SKILL_ID}))
        self.pipeline.handle_register_entity_spec(Message(
            "ovos.entity.register",
            {"skill_id": SKILL_ID, "entity_name": ENTITY_FILE, "lang": LANG,
             "samples": ENTITY_SAMPLES},
            {"skill_id": SKILL_ID}))

    def test_manifest_has_exactly_one_entry(self):
        self._dual_register()
        names = [e.get("name") for e in self.pipeline.registered_entities]
        self.assertEqual(names.count(CLEAN_NAME), 1)
        self.assertNotIn(MUNGED_NAME, names)

    def test_engine_has_exactly_one_object(self):
        self._dual_register()
        container = self.pipeline.containers[LANG]
        names = [e.name for e in container.entities.objects
                 + container.entities.objects_to_train]
        self.assertEqual(names.count(_wrapped(CLEAN_NAME)), 1)

    def test_spec_deregister_removes_the_legacy_twin(self):
        self._dual_register()
        self.pipeline.handle_deregister_entity_spec(Message(
            "ovos.entity.deregister",
            {"skill_id": SKILL_ID, "entity_name": ENTITY_FILE},
            {"skill_id": SKILL_ID}))
        names = [e.get("name") for e in self.pipeline.registered_entities]
        self.assertNotIn(CLEAN_NAME, names)


class TestEntityRemoval(unittest.TestCase):
    """Entities are stored as ``<skill_id>:{<entity>}``; removal must build
    the same key. Wrapping the whole namespaced name instead matched nothing,
    so entities were unremovable and re-registration stacked duplicates.
    """

    def setUp(self):
        self.pipeline = PadatiousPipeline(FakeBus())

    def tearDown(self):
        self.pipeline.shutdown()

    @staticmethod
    def _engine_names(container):
        return [e.name for e in container.entities.objects
                + container.entities.objects_to_train]

    def test_namespaced_entity_is_actually_removed_from_the_engine(self):
        container = self.pipeline.containers[LANG]
        container.add_entity(CLEAN_NAME, ENTITY_SAMPLES)
        self.assertIn(_wrapped(CLEAN_NAME), self._engine_names(container))

        container.remove_entity(CLEAN_NAME)

        self.assertNotIn(_wrapped(CLEAN_NAME), self._engine_names(container))

    def test_global_entity_removal_still_works(self):
        container = self.pipeline.containers[LANG]
        container.add_entity("weekday", ["monday", "tuesday"])
        self.assertIn("{weekday}", self._engine_names(container))

        container.remove_entity("weekday")

        self.assertNotIn("{weekday}", self._engine_names(container))

    def test_reregistering_an_entity_does_not_stack_duplicates(self):
        container = self.pipeline.containers[LANG]
        container.add_entity(CLEAN_NAME, ENTITY_SAMPLES)
        container.add_entity(CLEAN_NAME, ENTITY_SAMPLES + ["friday"])
        names = [e.name for e in container.entities.objects
                 + container.entities.objects_to_train]
        self.assertEqual(names.count(_wrapped(CLEAN_NAME)), 1)


if __name__ == "__main__":
    unittest.main()
