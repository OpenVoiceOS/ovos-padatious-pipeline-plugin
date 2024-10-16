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

from typing import Dict, List, Generator
from ovos_padatious.util import tokenize, expand_parentheses, remove_comments


class TrainData:
    """
    Represents training data for intent recognition, allowing for the management
    of collections of tokenized sentences from intent files.

    Attributes:
        sent_lists (Dict[str, List[List[str]]]): A dictionary mapping
        intent names to their corresponding tokenized sentence lists.
    """

    def __init__(self) -> None:
        self.sent_lists: Dict[str, List[List[str]]] = {}

    def add_lines(self, name: str, lines: List[str]) -> None:
        """
        Adds lines of text to the training data after processing.

        Args:
            name (str): The name of the intent for which lines are added.
            lines (List[str]): Lines of text to process.
        """
        lines = remove_comments(lines)
        self.sent_lists[name] = [
            sent for line in lines
            for sent in expand_parentheses(tokenize(line))
        ]
        self.sent_lists[name] = [sent for sent in self.sent_lists[name] if sent]

    def remove_lines(self, name: str) -> None:
        """
        Removes all lines associated with a given intent name.

        Args:
            name (str): The name of the intent to remove.
        """
        self.sent_lists.pop(name, None)

    def add_file(self, name: str, file_name: str) -> None:
        """
        Adds lines from a file to the training data.

        Args:
            name (str): The name of the intent for which lines are added.
            file_name (str): The name of the file containing lines of text.
        """
        with open(file_name, 'r', encoding='utf-8') as f:
            self.add_lines(name, f.readlines())

    def all_sents(self) -> Generator[List[str], None, None]:
        """
        Yields all sentences from all intents in the training data.

        Yields:
            Generator[List[str]]: A generator that produces tokenized sentences.
        """
        for sents in self.sent_lists.values():
            for sent in sents:
                yield sent

    def my_sents(self, my_name: str) -> Generator[List[str], None, None]:
        """
        Yields sentences for a specific intent name.

        Args:
            my_name (str): The name of the intent for which sentences are yielded.

        Yields:
            Generator[List[str]]: A generator that produces tokenized sentences for the specified intent.
        """
        for sent in self.sent_lists.get(my_name, []):
            yield sent

    def other_sents(self, my_name: str) -> Generator[List[str], None, None]:
        """
        Yields sentences for all intents except the specified one.

        Args:
            my_name (str): The name of the intent to exclude.

        Yields:
            Generator[List[str]]: A generator that produces tokenized sentences for all other intents.
        """
        for name, sents in self.sent_lists.items():
            if name != my_name:
                yield from sents
