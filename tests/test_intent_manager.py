# Copyright 2026 OpenVoiceOS
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import time
from concurrent.futures import ThreadPoolExecutor
from threading import Lock
from unittest.mock import MagicMock, patch

import pytest

from ovos_padatious.intent_manager import IntentManager


@patch("ovos_padatious.intent_manager.ThreadPoolExecutor")
def test_reuses_bounded_inference_executor(executor_class, tmp_path):
    executor = executor_class.return_value
    executor.map.return_value = []

    manager = IntentManager(str(tmp_path), max_workers=4)
    entity_manager = MagicMock()
    manager.calc_intents("first query", entity_manager)
    manager.calc_intents("second query", entity_manager)

    executor_class.assert_called_once_with(
        max_workers=4, thread_name_prefix="padatious-inference")
    assert executor.map.call_count == 2

    manager.shutdown(wait=False)
    executor.shutdown.assert_called_once_with(wait=False)


@pytest.mark.parametrize("workers", [0, -1, True, 1.5, "4"])
def test_rejects_invalid_inference_worker_count(tmp_path, workers):
    with pytest.raises(ValueError, match="positive integer"):
        IntentManager(str(tmp_path), max_workers=workers)


def test_worker_bound_applies_across_concurrent_queries(tmp_path):
    state = {"active": 0, "peak": 0}
    lock = Lock()

    class Match:
        def detokenize(self):
            pass

    class BlockingIntent:
        name = "blocking"

        def match(self, sent, entity_manager):
            with lock:
                state["active"] += 1
                state["peak"] = max(state["peak"], state["active"])
            time.sleep(0.01)
            with lock:
                state["active"] -= 1
            return Match()

    manager = IntentManager(str(tmp_path), max_workers=2)
    manager.objects = [BlockingIntent() for _ in range(8)]
    try:
        with ThreadPoolExecutor(max_workers=4) as callers:
            futures = [
                callers.submit(manager.calc_intents, "query", MagicMock())
                for _ in range(4)
            ]
            assert all(len(future.result()) == 8 for future in futures)
    finally:
        manager.shutdown()

    assert state["peak"] == 2
