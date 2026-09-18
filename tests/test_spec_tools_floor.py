"""The ovos-spec-tools floor names the release that exports every name this
package imports.

`ovos_padatious.opm` imports REGISTERED_TYPES, MalformedTypedSlots,
declared_slot_types, drop_unregistered_typed_slots and validate_typed_slots,
which ovos-spec-tools first exported in 1.11.0a1. A floor under that lets a
resolver install a pair that fails at import time. The first test imports the
names, so a CI job that installs the floor fails here and not deep in a suite;
the second reads the declared floor so the two cannot drift apart again."""
import re
import unittest
from pathlib import Path

try:
    import tomllib
except ImportError:  # python < 3.11
    tomllib = None

from packaging.version import Version

#: the ovos-spec-tools release that first exported every name below
FLOOR = Version("1.11.0a1")

NAMES = {
    "ovos_spec_tools": ("REGISTERED_TYPES", "MalformedTypedSlots",
                        "declared_slot_types", "drop_unregistered_typed_slots",
                        "validate_typed_slots", "SpecMessage", "closest_lang",
                        "expand", "standardize_lang", "gate_satisfied",
                        "context_slot_candidates"),
    "ovos_spec_tools.expansion": ("MalformedTemplate",),
}


class TestSpecToolsFloor(unittest.TestCase):
    def test_every_imported_name_is_exported(self):
        import importlib
        for module, names in NAMES.items():
            mod = importlib.import_module(module)
            for name in names:
                self.assertTrue(hasattr(mod, name),
                                f"{module} has no {name}: ovos-spec-tools is "
                                f"older than {FLOOR}")

    def test_declared_floor_is_not_under_the_release_that_exports_them(self):
        pyproject = Path(__file__).resolve().parents[1] / "pyproject.toml"
        text = pyproject.read_text(encoding="utf-8")
        specs = re.findall(r'"ovos-spec-tools([^"]*)"', text)
        self.assertTrue(specs, "pyproject.toml declares no ovos-spec-tools requirement")
        for spec in specs:
            m = re.search(r">=\s*([0-9A-Za-z.]+)", spec)
            self.assertIsNotNone(m, f"ovos-spec-tools{spec!r} has no >= floor")
            self.assertGreaterEqual(Version(m.group(1)), FLOOR,
                                    f"ovos-spec-tools{spec}: floor under {FLOOR}")
