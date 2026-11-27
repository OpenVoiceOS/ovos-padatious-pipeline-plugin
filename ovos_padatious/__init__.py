# Copyright 2017 Mycroft AI, Inc.
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

from .intent_container import IntentContainer
from .domain_container import DomainIntentContainer
from .match_data import MatchData

# The single source of truth is version.py, which pyproject.toml also
# reads for the distribution version. A literal here went stale at
# 0.4.8 while the package shipped 2.x, and training_manager.py salts
# every intent-cache hash with the major.minor of this name, so the
# salt could never move. The comment it carried named a setup.py that
# no longer exists.
from .version import __version__
