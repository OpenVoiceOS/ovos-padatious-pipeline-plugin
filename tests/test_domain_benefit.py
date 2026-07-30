"""Toy benchmark: an utterance the flat container misclassifies but
the DomainIntentContainer gets right.

Two skills share a "set X to Y" template surface. With every intent in
a single flat container, padatious sees ``set thermostat to N`` and
``set volume to N`` as near-duplicates and the global argmax can fall
on the wrong intent. Partitioning by skill domain narrows each
container's decision space; the per-domain top-1 plus global argmax
picks the right side.
"""
import unittest

from ovos_padatious.domain_container import DomainIntentContainer
from ovos_padatious.intent_container import IntentContainer


HOME = {
    "turn_on_lights":  ["turn on the lights", "lights on", "switch on the light"],
    "turn_off_lights": ["turn off the lights", "lights off", "switch off the light"],
    "set_thermostat":  ["set thermostat to 20 degrees",
                        "make it 22 degrees",
                        "set the temperature to 18"],
}

MEDIA = {
    "play_song":  ["play africa", "put on bohemian rhapsody", "play hey jude"],
    "pause_song": ["pause", "pause the music", "stop playback"],
    "set_volume": ["set volume to 5",
                   "make it 7 loud",
                   "set the volume to 3"],
}

# Utterance the flat engine often miscategorises because "set X to Y"
# matches set_thermostat at least as well as set_volume.
AMBIGUOUS = "set volume to 8"
EXPECTED_INTENT = "set_volume"


class TestDomainBeatsFlat(unittest.TestCase):
    def test_domain_picks_set_volume_when_flat_misroutes(self):
        flat = IntentContainer()
        for name, samples in {**HOME, **MEDIA}.items():
            flat.add_intent(name, samples)
        flat.train()
        flat_match = flat.calc_intent(AMBIGUOUS)

        domain = DomainIntentContainer()
        for name, samples in HOME.items():
            domain.add_domain_intent("home", name, samples)
        for name, samples in MEDIA.items():
            domain.add_domain_intent("media", name, samples)
        domain.train()
        domain_match = domain.calc_intent(AMBIGUOUS)

        # The whole point of the test: domain routing finds the right
        # intent here. We don't strictly require the flat container to
        # fail (padatious is sometimes good enough), but when the
        # difference shows up, the domain side must win.
        self.assertEqual(domain_match.name, EXPECTED_INTENT)
        if flat_match.name != EXPECTED_INTENT:
            self.assertNotEqual(flat_match.name, domain_match.name)


if __name__ == "__main__":
    unittest.main()
