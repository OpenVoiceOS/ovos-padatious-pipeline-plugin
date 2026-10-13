"""``_PatternCache.save`` writes under a temp name before ``os.replace``.
pid alone is not unique: two saves racing on the SAME path within one
process (two threads, or two calls racing on the same container) would
both pick the same temp name and race each other's os.replace -- the
losing writer's content is silently dropped, or its own file disappears
out from under it mid-write."""
import json
import os
import tempfile
import threading
import unittest
from unittest import mock

from ovos_padatious.padaos import _PatternCache


class TestPatternCacheTmpName(unittest.TestCase):
    def test_concurrent_saves_do_not_share_a_temp_name(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "patterns.json")
            caches = [_PatternCache(path) for _ in range(8)]
            for i, cache in enumerate(caches):
                cache._patterns = {f"line{i}": f"pattern{i}"}
                cache._used = {f"line{i}"}
                cache._dirty = True

            seen_tmp_names = []
            orig_replace = os.replace

            def recording_replace(src, dst):
                seen_tmp_names.append(src)
                orig_replace(src, dst)

            threads = [threading.Thread(target=c.save) for c in caches]
            with mock.patch("ovos_padatious.padaos.os.replace",
                           side_effect=recording_replace):
                for t in threads:
                    t.start()
                for t in threads:
                    t.join()

            self.assertEqual(len(seen_tmp_names), len(set(seen_tmp_names)),
                             "every concurrent save must pick its own, "
                             "never-reused temp name")
            self.assertTrue(os.path.exists(path),
                            "exactly one save's content survives at the "
                            "real path -- no leftover .tmp file beside it")
            leftover = [f for f in os.listdir(d) if f != "patterns.json"]
            self.assertEqual(leftover, [], f"no stray temp files: {leftover}")


if __name__ == "__main__":
    unittest.main()
