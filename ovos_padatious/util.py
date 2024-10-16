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

from xxhash import xxh32
from typing import List, Tuple, Dict, Any

# keep import for backwards compat
from ovos_utils.bracket_expansion import SentenceTreeParser, expand_parentheses


def lines_hash(lines: List[str]) -> bytes:
    """
    Creates a unique binary hash for a list of strings (lines).

    Args:
        lines (list<str>): List of strings that should be collectively hashed
    Returns:
        bytes: Binary hash of the given lines.
    """
    x = xxh32()
    for line in lines:
        x.update(line.encode())
    return x.digest()


def tokenize(sentence: str) -> List[str]:
    """
    Tokenizes a sentence into individual significant units (words, numbers, etc.).

    Args:
        sentence (str): The input sentence to tokenize, e.g., 'This is a sentence.'

    Returns:
        List[str]: List of tokens from the sentence, e.g., ['this', 'is', 'a', 'sentence'].
    """
    tokens: List[str] = []

    class Vars:
        start_pos = -1
        last_type = 'o'

    def update(c: str, i: int) -> None:
        if c.isalpha() or c in '-{}':
            t = 'a'  # alpha or special char (e.g. '-')
        elif c.isdigit() or c == '#':
            t = 'n'  # number
        elif c.isspace():
            t = 's'  # space
        else:
            t = 'o'  # other

        if t != Vars.last_type or t == 'o':
            if Vars.start_pos >= 0:
                token = sentence[Vars.start_pos:i].lower()
                if token not in '.!?':
                    tokens.append(token)
            Vars.start_pos = -1 if t == 's' else i
        Vars.last_type = t

    for i, char in enumerate(sentence):
        update(char, i)
    update(' ', len(sentence))  # finalize last token
    return tokens


def remove_comments(lines: List[str]) -> List[str]:
    """
    Removes comment lines from a list of strings.

    Args:
        lines (List[str]): List of lines that may contain comments (starting with '//').

    Returns:
        List[str]: Lines without any comments.
    """
    return [line for line in lines if not line.startswith('//')]


def resolve_conflicts(inputs: List[List[float]], outputs: List[List[float]]) -> Tuple[
    List[List[float]], List[List[float]]]:
    """
    Resolves conflicts in the input/output pairs by removing duplicates
    and combining output vectors for duplicate inputs.

    Args:
        inputs (List[List[float]]): List of input vectors.
        outputs (List[List[float]]): Corresponding list of output vectors.

    Returns:
        Tuple[List[List[float]], List[List[float]]]: The modified inputs and outputs
        with conflicts resolved (duplicates combined).
    """
    data: Dict[Tuple[float, ...], List[List[float]]] = {}

    for inp, out in zip(inputs, outputs):
        inp_tuple = tuple(inp)
        if inp_tuple in data:
            data[inp_tuple].append(out)
        else:
            data[inp_tuple] = [out]

    inputs_resolved: List[List[float]] = []
    outputs_resolved: List[List[float]] = []

    for inp, outs in data.items():
        inputs_resolved.append(list(inp))
        combined_out = [max(column[i] for column in outs) for i in range(len(outs[0]))]
        outputs_resolved.append(combined_out)

    return inputs_resolved, outputs_resolved


class StrEnum:
    """An enumeration class where the keys are strings."""

    @classmethod
    def values(cls) -> List[Any]:
        """
        Retrieves all values of the enum that are not special methods or attributes.

        Returns:
            List[Any]: List of values of the enum.
        """
        return [getattr(cls, attr) for attr in dir(cls)
                if not attr.startswith("__") and attr != 'values']
