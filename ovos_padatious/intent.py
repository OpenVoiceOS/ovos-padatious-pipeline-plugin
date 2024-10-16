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

import json
import math
from os.path import join
from typing import List, Optional

from ovos_padatious.match_data import MatchData
from ovos_padatious.pos_intent import PosIntent
from ovos_padatious.simple_intent import SimpleIntent
from ovos_padatious.trainable import Trainable


class Intent(Trainable):
    """
    Full intent object to handle entity extraction and intent matching.

    Attributes:
        simple_intent (SimpleIntent): An instance for handling simple intent matching.
        pos_intents (List[PosIntent]): A list of position intents associated with this intent.
    """

    def __init__(self, *args, **kwargs) -> None:
        super(Intent, self).__init__(*args, **kwargs)
        self.simple_intent: SimpleIntent = SimpleIntent(self.name)
        self.pos_intents: List[PosIntent] = []

    def match(self, sent: str, entities: Optional['EntityManager'] = None) -> MatchData:
        """
        Matches the input sentence against the intent and extracts entities.

        Args:
            sent (str): The input sentence to match.
            entities (Optional[EntityManager]): The entity manager containing known entities.

        Returns:
            MatchData: The best match data found, including intent name and confidence.
        """
        possible_matches: List[MatchData] = [MatchData(self.name, sent)]
        for pi in self.pos_intents:
            entity = entities.find(self.name, pi.token) if entities else None
            for i in list(possible_matches):
                possible_matches += pi.match(i, entity)

        possible_matches = [i for i in possible_matches if i.conf >= 0.0]

        for i in possible_matches:
            conf = ((i.conf / len(i.matches)) if len(i.matches) > 0 else 0) + 0.5
            i.conf = math.sqrt(conf * self.simple_intent.match(i.sent))

        return max(possible_matches, key=lambda x: x.conf)

    def save(self, folder: str) -> None:
        """
        Saves the intent data to the specified folder.

        Args:
            folder (str): The folder path to save the intent data.
        """
        prefix = join(folder, self.name)
        with open(prefix + '.hash', 'wb') as f:
            f.write(self.hash)
        self.simple_intent.save(prefix)
        prefix += '.pos'
        with open(prefix, 'w') as f:
            json.dump([i.token for i in self.pos_intents], f)
        for pos_intent in self.pos_intents:
            pos_intent.save(prefix)

    @classmethod
    def from_file(cls, name: str, folder: str) -> 'Intent':
        """
        Loads an intent object from the specified folder.

        Args:
            name (str): The name of the intent to load.
            folder (str): The folder path containing the intent data.

        Returns:
            Intent: The loaded intent object.
        """
        self = cls(name)
        prefix = join(folder, name)
        self.load_hash(prefix)
        self.simple_intent = SimpleIntent.from_file(name, prefix)
        prefix += '.pos'
        with open(prefix, 'r') as f:
            tokens = json.load(f)
        for token in tokens:
            self.pos_intents.append(PosIntent.from_file(prefix, token))
        return self

    def train(self, train_data: 'TrainData') -> None:
        """
        Trains the intent using the provided training data.

        Args:
            train_data (TrainData): The training data used for training the intent.
        """
        tokens = set(token for sent in train_data.my_sents(self.name) for token in sent if token.startswith('{'))
        self.pos_intents = [PosIntent(i, self.name) for i in tokens]

        self.simple_intent.train(train_data)
        for i in self.pos_intents:
            i.train(train_data)
