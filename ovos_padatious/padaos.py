import re
import time
from threading import Lock
from ovos_utils.log import LOG

#: Upper bound on the number of values inlined into a padaos entity
#: alternation. padaos builds one ``(a|b|c|...)`` regex per entity and
#: substitutes it verbatim into EVERY intent line that references the
#: slot, so the cost is O(values * referencing_lines): an auto-registered
#: entity with a couple thousand multi-word values (ovos-workshop can emit
#: exactly that), referenced from a few dozen intent lines, produces
#: megabyte regex sources whose compilation runs for minutes. Past the
#: cap the slot falls back to the same wildcard capture padaos already
#: uses for a ``{slot}`` with no registered entity at all; exact in-list
#: matching for large entities is still guaranteed end-to-end via the
#: neural tier's ``Entity.samples`` exact-match path, so nothing is lost
#: for listed values, only the padaos fast-path for out-of-cap entities.
PADAOS_ENTITY_INLINE_CAP = 64

#: Compiling a single intent line slower than this is a sign an inlined
#: entity is too large; logged as a warning naming the offender.
SLOW_COMPILE_WARN_SECONDS = 1.0


class _LineAlternationCapExceeded(Exception):
    """Raised by ``_cap_line_alternations`` when a literal intent-line
    alternation group exceeds PADAOS_ENTITY_INLINE_CAP; caught by
    ``_create_regex``, which skips the offending line for padaos."""
    def __init__(self, branch_count):
        self.branch_count = branch_count
        super().__init__(f"{branch_count} branches exceeds "
                          f"PADAOS_ENTITY_INLINE_CAP ({PADAOS_ENTITY_INLINE_CAP})")


class IntentContainer:
    def __init__(self):
        self.intent_lines, self.entity_lines = {}, {}
        self.intents, self.entities = {}, {}
        self.must_compile = True
        self.i = 0
        self.compile_lock = Lock()
        #: entity names skipped from inlining on the last compile because
        #: they exceeded PADAOS_ENTITY_INLINE_CAP; their slots fall back to
        #: an unverified wildcard capture, so callers must not treat a
        #: match against one of these slots as a verified in-list value.
        #: A literal intent-line alternation group over the same cap is
        #: NOT represented here - it has no registered entity backing it
        #: to verify against, so it is dropped from padaos entirely
        #: instead (see ``_cap_line_alternations``).
        self.capped_entities = set()

    def add_intent(self, name, lines):
        with self.compile_lock:
            self.must_compile = True
            self.intent_lines[name] = lines

    def remove_intent(self, name):
        with self.compile_lock:
            self.must_compile = True
            if name in self.intent_lines:
                del self.intent_lines[name]

    def add_entity(self, name, lines):
        with self.compile_lock:
            self.must_compile = True
            self.entity_lines[name] = lines

    def remove_entity(self, name):
        with self.compile_lock:
            self.must_compile = True
            if name in self.entity_lines:
                del self.entity_lines[name]

    def _create_pattern(self, line):
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

    #: matches a literal ``(a|b|c)`` alternation group written directly in
    #: an intent line's raw text, as long as it has no nested parentheses
    #: and isn't itself escaped (``\(...\)``); a group containing another
    #: group, or an escaped literal paren, is left alone entirely rather
    #: than risk miscounting or corrupting it.
    _RAW_ALTERNATION_RE = re.compile(r'(?<!\\)\(([^()]*)\)')

    def _cap_line_alternations(self, line):
        """
        Reject a literal ``(a|b|c|...)`` alternation group written
        directly in an intent line once it exceeds PADAOS_ENTITY_INLINE_CAP
        branches, checked on the RAW line before the expensive
        ``_create_pattern`` rewrite pipeline runs several regex passes
        over it (capping only the resulting regex, after the pipeline
        already ran over the huge raw text, does not save the cost).
        padaos otherwise turns such a group into a plain regex alternation
        with no bound at all, unlike registered entities (capped in
        ``_compile``); a generated or pathological line with hundreds or
        thousands of branches, or repeated across many lines, reproduces
        the same compile blowup PADAOS_ENTITY_INLINE_CAP was introduced to
        fix.

        Unlike an over-cap ENTITY slot - which still falls back to a
        wildcard capture safely, because ``_padaos_entities_verified``
        can check the matched text against that entity's own sample list
        before trusting it - a literal line group has no registered
        entity behind it at all. A wildcard standing in for the group
        makes the WHOLE LINE match almost any utterance containing the
        surrounding words, with nothing to verify the guess against; an
        earlier version of this fix did exactly that and a wildcard-line
        match for an unrelated intent went uncaught. So an over-cap line
        group raises instead, and ``_create_regex`` treats it the same
        way a malformed line is already treated: skip the line entirely
        (this intent's other lines still register; see ``_create_regex``).

        Raises:
            _LineAlternationCapExceeded: if a group exceeds the cap.
        """
        def repl(match):
            branches = re.split(r'(?<!\\)\|', match.group(1))
            if len(branches) > PADAOS_ENTITY_INLINE_CAP:
                raise _LineAlternationCapExceeded(len(branches))
            return match.group(0)
        return self._RAW_ALTERNATION_RE.sub(repl, line)

    def _create_intent_pattern(self, line, intent_name):
        namespace = intent_name.split(':')[0] + ':'
        line = self._cap_line_alternations(line)
        line = self._create_pattern(line)
        replacements = {}
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

    def _create_regex(self, line, intent_name):
        """ Create regex and return. If error occurs returns None. """
        try:
            return re.compile(self._create_intent_pattern(line, intent_name),
                              re.IGNORECASE)
        except _LineAlternationCapExceeded as e:
            # same treatment as a malformed line (see util.expand_or_skip):
            # log and contribute no padaos pattern for it. The neural tier
            # still trains on this line's expanded samples independently
            # (subject to its own caps), so matching is not lost outright,
            # only padaos' exact-template fast path for this one line.
            LOG.warning(
                f"intent '{intent_name}' line has a literal alternation "
                f"group with {e.branch_count} branches (cap is "
                f"{PADAOS_ENTITY_INLINE_CAP}); skipping it for padaos: "
                f"{line!r}"
            )
            return None
        except Exception as e:
            LOG.exception(f'Failed to parse the line "{line}" for {intent_name}')
            return None

    def create_regexes(self, lines, intent_name):
        regexes = [self._create_regex(line, intent_name)
                   for line in sorted(lines, key=len, reverse=True)
                   if line.strip()]
        # Filter out all regexes that fails
        return [r for r in regexes if r is not None]

    def compile(self):
        with self.compile_lock:
            self._compile()

    def _compile(self):
        start = time.monotonic()
        largest_entity, largest_size = None, 0
        self.entities = {}
        self.capped_entities = set()
        for ent_name, lines in self.entity_lines.items():
            values = [line for line in lines if line.strip()]
            if len(values) > largest_size:
                largest_entity, largest_size = ent_name, len(values)
            if len(values) > PADAOS_ENTITY_INLINE_CAP:
                self.capped_entities.add(ent_name)
                # too many values to inline: skip so referencing slots fall
                # back to the plain wildcard capture (see PADAOS_ENTITY_INLINE_CAP)
                continue
            self.entities[ent_name] = r'({})'.format('|'.join(
                self._create_pattern(line) for line in values
            ))
        self.intents = {
            intent_name: self.create_regexes(lines, intent_name)
            for intent_name, lines in self.intent_lines.items()
        }
        self.must_compile = False
        duration = time.monotonic() - start
        if duration > SLOW_COMPILE_WARN_SECONDS:
            if largest_entity is None:
                LOG.warning(f"padaos compile took {duration:.2f}s; "
                            f"no entities registered")
            else:
                # this reports the FULL container compile (every entity and
                # intent line), not the cost of the named entity alone: an
                # entity past PADAOS_ENTITY_INLINE_CAP is skipped from
                # inlining (cheap regardless of its value count), so a slow
                # compile with a capped "largest entity" means the time is
                # coming from the sheer number of other entities/intents in
                # the container, not from this one
                capped = largest_entity in self.capped_entities
                LOG.warning(
                    f"padaos compile took {duration:.2f}s for the full "
                    f"container ({len(self.entity_lines)} entities, "
                    f"{len(self.intent_lines)} intents); largest entity "
                    f"'{largest_entity}' has {largest_size} values"
                    f"{' (capped, not inlined)' if capped else ''}"
                )

    def _calc_entities(self, query, regexes):
        for regex in regexes:
            match = regex.match(query)
            if match:
                yield {
                    k.rsplit('__', 1)[0].replace('__colon__', ':'): v.strip()
                    for k, v in match.groupdict().items() if v
                }

    def calc_intents(self, query):
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

    def calc_intent(self, query):
        return min(
            self.calc_intents(query),
            key=lambda x: sum(map(len, x['entities'].values())),
            default={'name': None, 'entities': {}}
        )
