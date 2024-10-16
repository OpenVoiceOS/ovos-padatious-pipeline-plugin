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

from typing import List, Optional
from fann2 import libfann as fann

from ovos_padatious.id_manager import IdManager
from ovos_padatious.util import resolve_conflicts, StrEnum


class Ids(StrEnum):
    """Enumeration for token IDs used in intent matching."""
    unknown_tokens = ':0'
    w_1 = ':1'
    w_2 = ':2'
    w_3 = ':3'
    w_4 = ':4'


class SimpleIntent:
    """General intent used to match sentences or phrases."""
    LENIENCE: float = 0.6

    def __init__(self, name: str = '') -> None:
        self.name: str = name
        self.ids: IdManager = IdManager(Ids)
        self.net: Optional[fann.neural_net] = None

    def match(self, sent: List[str]) -> float:
        """Matches a sentence against the intent.

        Args:
            sent (List[str]): The input sentence as a list of tokens.

        Returns:
            float: The confidence score of the match, between 0 and 1.
        """
        return max(0, self.net.run(self.vectorize(sent))[0])

    def vectorize(self, sent: List[str]) -> List[float]:
        """Converts a sentence into a vector representation.

        Args:
            sent (List[str]): The input sentence as a list of tokens.

        Returns:
            List[float]: The vector representation of the sentence.
        """
        vector = self.ids.vector()
        unknown = 0
        for token in sent:
            if token in self.ids:
                self.ids.assign(vector, token, 1.0)
            else:
                unknown += 1
        if sent:
            self.ids.assign(vector, Ids.unknown_tokens, unknown / len(sent))
            self.ids.assign(vector, Ids.w_1, len(sent) / 1)
            self.ids.assign(vector, Ids.w_2, len(sent) / 2.0)
            self.ids.assign(vector, Ids.w_3, len(sent) / 3.0)
            self.ids.assign(vector, Ids.w_4, len(sent) / 4.0)
        return vector

    def configure_net(self) -> None:
        """Configures the neural network for intent matching."""
        self.net = fann.neural_net()
        self.net.create_standard_array([len(self.ids), 10, 1])
        self.net.set_activation_function_hidden(fann.SIGMOID_SYMMETRIC_STEPWISE)
        self.net.set_activation_function_output(fann.SIGMOID_SYMMETRIC_STEPWISE)
        self.net.set_train_stop_function(fann.STOPFUNC_BIT)
        self.net.set_bit_fail_limit(0.1)

    def train(self, train_data) -> None:
        """Trains the intent matching neural network.

        Args:
            train_data: The training data containing sentences and outputs.
        """
        for sent in train_data.my_sents(self.name):
            self.ids.add_sent(sent)

        inputs: List[List[float]] = []
        outputs: List[List[float]] = []

        def add(vec: List[str], out: float) -> None:
            """Adds a vector and its corresponding output to the training data."""
            inputs.append(self.vectorize(vec))
            outputs.append([out])

        def pollute(sent: List[str], p: int) -> None:
            """Pollutes the sentence with null tokens for training."""
            sent = sent[:]
            for _ in range(int((len(sent) + 2) / 3)):
                sent.insert(p, ':null:')
            add(sent, self.LENIENCE)

        def weight(sent: List[str]) -> None:
            """Calculates and adds weights for the words in a sentence."""
            def calc_weight(w: str) -> float:
                return pow(len(w), 3.0)

            total_weight = sum(calc_weight(word) for word in sent)
            for word in sent:
                word_weight = 0 if word.startswith('{') else calc_weight(word)
                add([word], word_weight / total_weight)

        for sent in train_data.my_sents(self.name):
            add(sent, 1.0)
            weight(sent)

            # Generate samples with extra unknown tokens unless
            # the sentence is supposed to allow unknown tokens via the special :0
            if not any(word[0] == ':' and word != ':' for word in sent):
                pollute(sent, 0)
                pollute(sent, len(sent))

        for sent in train_data.other_sents(self.name):
            add(sent, 0.0)
        add([':null:'], 0.0)
        add([], 0.0)

        for sent in train_data.my_sents(self.name):
            without_entities = sent[:]
            for i, token in enumerate(without_entities):
                if token.startswith('{'):
                    without_entities[i] = ':null:'
            if without_entities != sent:
                add(without_entities, 0.0)

        inputs, outputs = resolve_conflicts(inputs, outputs)

        train_data_instance = fann.training_data()
        train_data_instance.set_train_data(inputs, outputs)

        for _ in range(10):
            self.configure_net()
            self.net.train_on_data(train_data_instance, 1000, 0, 0)
            self.net.test_data(train_data_instance)
            if self.net.get_bit_fail() == 0:
                break

    def save(self, prefix: str) -> None:
        """Saves the intent and its neural network to files.

        Args:
            prefix (str): The prefix for the saved files.
        """
        prefix += '.intent'
        self.net.save(f"{prefix}.net")  # Must have str()
        self.ids.save(prefix)

    @classmethod
    def from_file(cls, name: str, prefix: str) -> 'SimpleIntent':
        """Creates an instance of SimpleIntent from a saved file.

        Args:
            name (str): The name of the intent.
            prefix (str): The prefix for the saved files.

        Returns:
            SimpleIntent: The loaded SimpleIntent instance.
        """
        prefix += '.intent'
        instance = cls(name)
        instance.net = fann.neural_net()
        if not instance.net.create_from_file(f"{prefix}.net"):  # Must have str()
            raise FileNotFoundError(f"{prefix}.net")
        instance.ids.load(prefix)
        return instance
