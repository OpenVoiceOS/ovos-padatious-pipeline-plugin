"""The package version has ONE source, and the cache salt follows it.

`training_manager.py` salts every intent-cache hash with the major.minor of
`ovos_padatious.__version__`, so that a cache built by an older engine is
retrained instead of loaded. A literal in `__init__.py` froze that name at
'0.4.8' while the package shipped 2.x, which pinned the salt at '0.4' for
every install and every future release.
"""

from os.path import splitext

import ovos_padatious
from ovos_padatious.version import __version__ as version_py


def test_package_version_comes_from_version_py():
    """The name the salt reads is the one the build reads."""
    assert ovos_padatious.__version__ == version_py


def test_package_version_matches_installed_metadata():
    """pyproject builds the distribution from version.py, so an install
    disagreeing with the imported name means a stale literal is back."""
    from importlib.metadata import version as dist_version
    assert ovos_padatious.__version__ == dist_version("ovos-padatious")


def test_version_is_not_the_frozen_literal():
    """The exact value that was frozen, named so a re-freeze is loud."""
    assert ovos_padatious.__version__ != "0.4.8"


def test_cache_salt_tracks_major_minor():
    """What training_manager.py actually computes.

    The salt must be the running engine's major.minor. It is allowed to
    stay put across a patch release and must move on a minor or major one.
    """
    salt = splitext(ovos_padatious.__version__)[0]
    major_minor = ".".join(version_py.split(".")[:2])
    assert salt == major_minor
    assert salt != "0.4", "the cache salt is frozen at the 0.4.8 literal"
