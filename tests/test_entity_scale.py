"""Entity training must stay bounded and deterministic for large value sets.

Auto-registered ``.entity`` files (ovos-workshop 9.5.0a1+) can carry thousands
of values; the engine owns choosing a subset it can consume. These tests pin:

* a listed value scores exactly 1.0 through ``Entity.match`` (deterministic
  exact path, no net variance);
* the net's vocabulary stays bounded no matter how large the entity file is;
* the strided subset pick is deterministic and keeps both endpoints;
* samples survive a cache round trip.
"""
import shutil
import tempfile
import unittest

from ovos_padatious import IntentContainer
from ovos_padatious.util import tokenize
from ovos_padatious.entity import (Entity, ENTITY_NET_TRAINING_CAP,
                                   _strided_subset)


class TestStridedSubset(unittest.TestCase):
    def test_under_cap_is_identity(self):
        sents = [(str(i),) for i in range(10)]
        self.assertEqual(_strided_subset(sents, 256), set(sents))

    def test_over_cap_bounds_and_keeps_endpoints(self):
        sents = [(f"{i:05d}",) for i in range(5000)]
        picked = _strided_subset(sents, 256)
        self.assertEqual(len(picked), 256)
        self.assertIn(sents[0], picked)
        self.assertIn(sents[-1], picked)

    def test_deterministic(self):
        sents = [(f"{i:05d}",) for i in range(3000)]
        self.assertEqual(_strided_subset(sents, 256),
                         _strided_subset(sents, 256))


class TestLargeEntityTraining(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.container = IntentContainer(self.tmp)
        self.values = [f"thing{i:04d}" for i in range(2000)]
        self.container.add_entity("item", self.values)
        self.container.add_intent("get", ["fetch {item} now", "grab {item}"])
        self.container.train(debug=False, single_thread=True)
        self.entity = self.container.entities.entity_dict.get("{item}")

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_listed_value_scores_exactly_one(self):
        self.assertIsNotNone(self.entity)
        for value in ("thing0000", "thing1234", "thing1999"):
            self.assertEqual(self.entity.match(tokenize(value)), 1.0)

    def test_all_values_kept_as_samples(self):
        self.assertEqual(len(self.entity.samples), len(self.values))

    def test_net_vocabulary_is_bounded(self):
        # vocab comes from at most ENTITY_NET_TRAINING_CAP positives (plus
        # the handful of structural ids), never from all 2000 values
        self.assertLessEqual(len(self.entity.ids),
                             ENTITY_NET_TRAINING_CAP + 32)

    def test_unlisted_value_still_scored_by_net(self):
        conf = self.entity.match(["somethingelse"])
        self.assertGreaterEqual(conf, 0.0)
        self.assertLess(conf, 1.0)

    def test_samples_survive_cache_roundtrip(self):
        fresh = IntentContainer(self.tmp)
        fresh.add_entity("item", self.values)
        fresh.add_intent("get", ["fetch {item} now", "grab {item}"])
        fresh.train(debug=False, single_thread=True)  # cache hit, no retrain
        ent = fresh.entities.entity_dict.get("{item}")
        self.assertEqual(ent.match(tokenize("thing1234")), 1.0)
        self.assertEqual(len(ent.samples), len(self.values))




class TestCacheFormatUpgrade(unittest.TestCase):
    def test_pre_sidecar_cache_retrains_once(self):
        import glob
        from os.path import join
        from ovos_padatious.util import lines_hash
        tmp = tempfile.mkdtemp()
        c = IntentContainer(tmp)
        c.add_entity("item", ["london", "porto"])
        c.add_intent("go", ["travel to {item}"])
        c.train(debug=False)
        sidecars = glob.glob(join(tmp, "*.samples"))
        self.assertTrue(sidecars, "training must write the samples sidecar")
        # simulate a pre-format2 cache: old-style hash, no sidecar
        import ovos_padatious
        from os.path import splitext
        min_ver = splitext(ovos_padatious.__version__)[0]
        for f in sidecars:
            import os
            os.remove(f)
        ent_hash = [f for f in glob.glob(join(tmp, "*.hash")) if "item" in f][0]
        with open(ent_hash, "wb") as f:
            f.write(lines_hash([min_ver] + ["london", "porto"]))
        fresh = IntentContainer(tmp)
        fresh.add_entity("item", ["london", "porto"])
        fresh.add_intent("go", ["travel to {item}"])
        fresh.train(debug=False)
        self.assertTrue(glob.glob(join(tmp, "*.samples")),
                        "stale-format cache must retrain and restore the sidecar")
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
