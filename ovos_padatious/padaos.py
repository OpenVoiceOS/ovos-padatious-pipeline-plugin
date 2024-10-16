import re
from threading import Lock
from typing import Dict, List, Optional, Union
from ovos_utils.log import LOG


class IntentContainer:
    def __init__(self) -> None:
        """
        Initializes the IntentContainer with empty intent and entity lines,
        a flag for compilation, and a thread lock.
        """
        self.intent_lines: Dict[str, List[str]] = {}
        self.entity_lines: Dict[str, List[str]] = {}
        self.intents: Dict[str, List[re.Pattern]] = {}
        self.entities: Dict[str, str] = {}
        self.must_compile: bool = True
        self.i: int = 0
        self.compile_lock: Lock = Lock()

    def add_intent(self, name: str, lines: List[str]) -> None:
        """
        Adds an intent with the given name and lines to the container.

        Args:
            name (str): The name of the intent.
            lines (List[str]): The lines associated with the intent.
        """
        with self.compile_lock:
            self.must_compile = True
            self.intent_lines[name] = lines

    def remove_intent(self, name: str) -> None:
        """
        Removes the intent with the specified name from the container.

        Args:
            name (str): The name of the intent to remove.
        """
        with self.compile_lock:
            self.must_compile = True
            self.intent_lines.pop(name, None)

    def add_entity(self, name: str, lines: List[str]) -> None:
        """
        Adds an entity with the given name and lines to the container.

        Args:
            name (str): The name of the entity.
            lines (List[str]): The lines associated with the entity.
        """
        with self.compile_lock:
            self.must_compile = True
            self.entity_lines[name] = lines

    def remove_entity(self, name: str) -> None:
        """
        Removes the entity with the specified name from the container.

        Args:
            name (str): The name of the entity to remove.
        """
        with self.compile_lock:
            self.must_compile = True
            self.entity_lines.pop(name, None)

    def _create_pattern(self, line: str) -> str:
        """
        Creates a regex pattern from the provided line.

        Args:
            line (str): The line to convert into a regex pattern.

        Returns:
            str: The generated regex pattern.
        """
        for pat, rep in (
                # === Preserve Plain Parentheses ===
                (r'\(([^\|)]*)\)', r'{~(\1)~}'),  # (hi) -> {~(hi)~}

                # === Convert to regex literal ===
                (r'(\W)', r'\\\1'),
                (r' {} '.format, None),  # 'abc' -> ' abc '

                # === Unescape Chars for Convenience ===
                (r'\\ ', r' '),  # "\ " -> " "
                (r'\\{', r'{'),  # \{ -> {
                (r'\\}', r'}'),  # \} -> }
                (r'\\#', r'#'),  # \# -> #

                # === Support Parentheses Expansion ===
                (r'(?<!\\{\\~)\\\(', r'(?:'),  # \( -> (  ignoring  \{\~\(
                (r'\\\)(?!\\~\\})', r')'),  # \) -> )  ignoring  \)\~\}
                (r'\\{\\~\\\(', r'\\('),  # \{\~\( -> \(
                (r'\\\)\\~\\}', r'\\)'),  # \)\~\}  -> \)
                (r'\\\|', r'|'),  # \| -> |

                # === Support Special Symbols ===
                (r'(?<=\s)\\:0(?=\s)', r'\\w+'),
                (r'#', r'\\d'),
                (r'\d', r'\\d'),

                # === Space Word Separations ===
                (r'(?<!\\)(\w)([^\w\s}])', r'\1 \2'),  # a:b -> a :b
                (r'([^\\\w\s{])(\w)', r'\1 \2'),  # a :b -> a : b

                # === Make Symbols Optional ===
                (r'(\\[^\w ])', r'\1?'),

                # === Force 1+ Space Between Words ===
                (r'(?<=(\w|\}))(\\\s|\s)+(?=\S)', r'\\W+'),

                # === Force 0+ Space Between Everything Else ===
                (r'\s+', r'\\W*'),
        ):
            if callable(pat):
                line = pat(line)
            else:
                line = re.sub(pat, rep, line)
        return line

    def _create_intent_pattern(self, line: str, intent_name: str) -> str:
        """
        Creates an intent-specific regex pattern from the line.

        Args:
            line (str): The line to create a pattern for.
            intent_name (str): The name of the intent.

        Returns:
            str: The generated regex pattern for the intent.
        """
        namespace = intent_name.split(':')[0] + ':'
        line = self._create_pattern(line)
        replacements: Dict[str, str] = {}

        for ent_name in set(re.findall(r'{([a-z_:]+)}', line)):
            replacements[ent_name] = r'(?P<{}__{{}}>.*?\w.*?)'.format(ent_name)

        for ent_name, ent in self.entities.items():
            ent_regex = r'(?P<{}__{{}}>{})'
            if ent_name.startswith(namespace):
                replacements[ent_name[len(namespace):]] = ent_regex.format(
                    ent_name[len(namespace):], ent
                )
            else:
                replacements[ent_name] = ent_regex.format(ent_name.replace(':', '__colon__'), ent)

        for key, value in replacements.items():
            line = line.replace('{' + key + '}', value.format(self.i), 1)
            self.i += 1

        return '^{}$'.format(line)

    def _create_regex(self, line: str, intent_name: str) -> Optional[re.Pattern]:
        """
        Creates a regex pattern and returns it. If an error occurs, returns None.

        Args:
            line (str): The line to compile into a regex.
            intent_name (str): The name of the intent.

        Returns:
            Optional[re.Pattern]: The compiled regex pattern or None if an error occurs.
        """
        try:
            return re.compile(self._create_intent_pattern(line, intent_name), re.IGNORECASE)
        except Exception as e:
            LOG.exception(f'Failed to parse the line "{line}" for {intent_name}')
            return None

    def create_regexes(self, lines: List[str], intent_name: str) -> List[Optional[re.Pattern]]:
        """
        Creates a list of regex patterns from the provided lines.

        Args:
            lines (List[str]): The lines to create regex patterns from.
            intent_name (str): The name of the intent.

        Returns:
            List[Optional[re.Pattern]]: A list of compiled regex patterns, excluding None values.
        """
        regexes = [self._create_regex(line, intent_name)
                   for line in sorted(lines, key=len, reverse=True)
                   if line.strip()]
        # Filter out all regexes that fails
        return [r for r in regexes if r is not None]

    def compile(self) -> None:
        """
        Compiles the intent and entity regex patterns. Should be called when the state changes.
        """
        with self.compile_lock:
            self._compile()

    def _compile(self) -> None:
        """Internal method to compile entity and intent patterns."""
        self.entities = {
            ent_name: r'({})'.format('|'.join(
                self._create_pattern(line) for line in lines if line.strip()
            ))
            for ent_name, lines in self.entity_lines.items()
        }
        self.intents = {
            intent_name: self.create_regexes(lines, intent_name)
            for intent_name, lines in self.intent_lines.items()
        }
        self.must_compile = False

from typing import Dict, List, Optional, Union, Iterator

    def _calc_entities(self, query: str, regexes: List[re.Pattern]) -> Iterator[Dict[str, str]]:
        """
        Calculates entities from a given query using the provided regex patterns.

        Args:
            query (str): The query to extract entities from.
            regexes (List[re.Pattern]): The list of regex patterns to use for extraction.

        Yields:
            Dict[str, str]: A dictionary of extracted entities.
        """
        for regex in regexes:
            match = regex.match(query)
            if match:
                yield {
                    k.rsplit('__', 1)[0].replace('__colon__', ':'): v.strip()
                    for k, v in match.groupdict().items() if v
                }

from typing import Dict, List, Optional, Union, Iterator

    def calc_intents(self, query: str) -> Iterator[Dict[str, Union[str, Dict[str, str]]]]:
        """
        Calculates intents for a given query.

        Args:
            query (str): The input query to process.

        Yields:
            Dict[str, Union[str, Dict[str, str]]]: A dictionary containing intent names and entities.
        """
        query = ' ' + query + ' '
        if self.must_compile:
            self.compile()
        for intent_name, regexes in self.intents.items():
            entities = list(self._calc_entities(query, regexes))
            if entities:
                yield {
                    'name': intent_name,
                    'entities': min(entities, key=lambda x: sum(map(len, x.values())))
                }

    def calc_intent(self, query: str) -> Dict[str, Union[str, Dict[str, str]]]:
        """
        Calculates the most likely intent for a given query.

        Args:
            query (str): The input query to process.

        Returns:
            Dict[str, Union[str, Dict[str, str]]]: A dictionary containing the intent name and entities.
        """
        return min(
            self.calc_intents(query),
            key=lambda x: sum(map(len, x['entities'].values())),
            default={'name': None, 'entities': {}}
        )
