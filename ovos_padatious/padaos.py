import json
import os
import re
import time
import uuid
from itertools import count
from typing import Dict, Optional
from threading import Lock, get_ident
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


class _IntentRegexes(list):
    """The list of an intent's compiled line regexes, carrying a single
    ``prefilter`` regex (the alternation of them all) used as a fast reject
    on the query path.

    It is a plain ``list`` in every observable way - indexing, ``len``,
    equality to ``[]`` - so it is a drop-in for the bare list
    ``self.intents[name]`` used to hold, while binding the prefilter to the
    regexes it was built from. Because the two travel together in one
    object, a single ``self.intents`` swap publishes both atomically: a
    lock-free reader in ``calc_intents`` can never pair one compile's
    regexes with another compile's prefilter.

    The prefilter is not compiled with the line regexes: compiling it is a
    second full regex compile of every line and it only speeds up queries.
    A background warm builds it after a compile; queries use it once it is
    there (``built_prefilter``) and scan the lines until then. At boot that
    halved the cost of every padaos compile.
    Two threads racing on the first use both compile the same pattern and
    store equal regexes, so no lock is needed.
    """
    __slots__ = ('_prefilter', 'plain')

    _UNBUILT = object()

    def __init__(self, regexes=(), plain=frozenset()):
        super().__init__(regexes)
        self._prefilter = self._UNBUILT
        #: the intent's plain lines (words and spaces only) as lowercased
        #: word tuples; see ``IntentContainer._plain_words``
        self.plain = plain

    @property
    def prefilter(self):
        prefilter = self._prefilter
        if prefilter is self._UNBUILT:
            if not self:
                prefilter = None
            elif len(self) == 1:
                prefilter = self[0]
            else:
                prefilter = re.compile(
                    '|'.join('(?:{})'.format(r.pattern) for r in self),
                    re.IGNORECASE)
            self._prefilter = prefilter
        return prefilter

    @prefilter.setter
    def prefilter(self, value):
        self._prefilter = value

    @property
    def built_prefilter(self):
        """The prefilter if something already built it, else None. The query
        path reads this one: building it costs a second compile of every
        line, which a query that just compiled its language (or one asked
        before the background warm got to it) should not pay; the scan
        without it gives the same answer."""
        prefilter = self._prefilter
        return None if prefilter is self._UNBUILT else prefilter


#: Bumped whenever ``_create_pattern``'s rewrite rules change, so a pattern
#: cache written by older rules is ignored rather than trusted.
PATTERN_RULES_VERSION = 1

#: A template line of words separated by spaces and plain symbols, and
#: nothing else: no digit (``_create_pattern`` turns each into ``\d``), no
#: slot or entity (``{}``), no group or alternation (``()|``), no ``#``,
#: ``:``, ``\`` or ``~`` (each has its own rewrite rule). ``_create_pattern``
#: makes such a symbol optional and puts ``\W+`` between words, so the line
#: matches a query exactly when their words, split on non-word characters,
#: are the same (case-insensitively).
_PLAIN_SYMBOL = r'[^\w\s{}()|#:\\~]'
_PLAIN_LINE = re.compile(r'^(?:\s|{s})*[^\W\d]+(?:(?:\s|{s})+[^\W\d]+)*(?:\s|{s})*$'.format(
    s=_PLAIN_SYMBOL))
_NON_WORD = re.compile(r'\W+')


def _words(text):
    """``text`` split on runs of non-word characters, each word lowercased.

    Split before lowercasing: ``str.lower()`` turns a dotted capital I
    ("İstanbul") into "i" plus a combining dot, and a combining mark is not a
    word character. The dotted I folds to a plain "i", as the regex's
    IGNORECASE matching folds it."""
    return tuple(w.replace('İ', 'i').lower() for w in _NON_WORD.split(text) if w)


class _PatternCache:
    """``line -> _create_pattern(line)``, optionally persisted to a JSON file.

    Rewriting a template line into its regex source runs about twenty regex
    substitutions over it, and was two thirds of a padaos compile; the
    result depends on the line alone. A container given a file reads it on
    first use and writes it back after a compile that added lines, so a
    restart with unchanged skills rewrites nothing. A file that cannot be
    read, or was written by other rules, is ignored.
    """

    def __init__(self, path: Optional[str] = None):
        self.path = path
        self._patterns: Optional[Dict[str, str]] = None
        self._used = set()
        self._dirty = False

    def _load(self) -> Dict[str, str]:
        if self._patterns is None:
            self._patterns = {}
            if self.path:
                try:
                    with open(self.path, encoding="utf-8") as f:
                        data = json.load(f)
                    if data.get("rules") == PATTERN_RULES_VERSION:
                        self._patterns = {str(k): str(v) for k, v in data["patterns"].items()}
                except (OSError, ValueError, KeyError, AttributeError, TypeError):
                    self._patterns = {}
        return self._patterns

    def get(self, line: str, create) -> str:
        patterns = self._load()
        self._used.add(line)
        pattern = patterns.get(line)
        if pattern is None:
            pattern = patterns[line] = create(line)
            self._dirty = True
        return pattern

    def save(self) -> None:
        if not (self.path and self._dirty and self._patterns is not None):
            return
        # pid alone collides within one process: two threads (or two calls
        # racing on the same container) saving the same path at once would
        # both write the same temp name and race each other's os.replace.
        tmp = f"{self.path}.{os.getpid()}.{get_ident()}.{uuid.uuid4().hex[:8]}.tmp"
        try:
            with open(tmp, "w", encoding="utf-8") as f:
                # only the lines this process compiled: retired lines drop out
                json.dump({"rules": PATTERN_RULES_VERSION,
                           "patterns": {line: self._patterns[line] for line in self._used
                                        if line in self._patterns}}, f)
            os.replace(tmp, self.path)
            self._dirty = False
        except OSError as e:
            LOG.debug(f"padaos pattern cache not written to {self.path}: {e}")
            try:
                os.remove(tmp)
            except OSError:
                pass


class IntentContainer:
    def __init__(self, pattern_cache: Optional[str] = None):
        """
        Args:
            pattern_cache: optional file keeping each template line's regex
                source across restarts (see ``_PatternCache``).
        """
        self._patterns = _PatternCache(pattern_cache)
        self.intent_lines, self.entity_lines = {}, {}
        self.intents, self.entities = {}, {}
        self.must_compile = True
        self.compile_lock = Lock()
        #: bumped by every registration; ``_compile`` compares it against
        #: its own snapshot to tell whether a registration landed while it
        #: was compiling off-lock.
        self._mutation_gen = 0
        #: entity names skipped from inlining on the last compile because
        #: they exceeded PADAOS_ENTITY_INLINE_CAP; their slots fall back to
        #: an unverified wildcard capture, so callers must not treat a
        #: match against one of these slots as a verified in-list value.
        #: A literal intent-line alternation group over the same cap is
        #: NOT represented here - it has no registered entity backing it
        #: to verify against, so it is dropped from padaos entirely
        #: instead (see ``_cap_line_alternations``).
        self.capped_entities = set()
        # one-shot guard so a pending compile logs at most once per
        # dirty->compiled cycle instead of once per query (see calc_intents)
        self._warned_pending_compile = False
        #: per-name memo of the last compile's output, so a compile only
        #: rebuilds what a registration actually changed. Each entry is
        #: ``(key, product)``: an entity's key is its lines, an intent's key
        #: is its lines plus the inlined patterns of exactly the entities
        #: its lines can reference. Only ``_compile`` reads or writes these,
        #: and compiles are serialized by their caller (see ``compile``).
        self._entity_cache = {}
        self._intent_cache = {}

    def add_intent(self, name, lines):
        with self.compile_lock:
            self.must_compile = True
            self._mutation_gen += 1
            self.intent_lines[name] = lines
            # Replacement is symmetric with removal: re-registering an
            # existing name with different samples must not keep matching
            # the RETIRED compiled regex - with its now-stale slot
            # captures - until some future background compile happens to
            # run. Drop the old compiled entry immediately; the new
            # content is visible once that compile actually lands, same
            # as any other addition.
            self.intents.pop(name, None)

    def remove_intent(self, name):
        with self.compile_lock:
            self.must_compile = True
            self._mutation_gen += 1
            if name in self.intent_lines:
                del self.intent_lines[name]
            # Drop the already-compiled entry immediately too: the match
            # path (calc_intents) never compiles anymore, so a removal
            # that only dirtied must_compile would keep matching against
            # the stale compiled regex until some future background
            # compile happens to run. Deregistration/detach is a runtime
            # gate, not a compile product, and must take effect now.
            self.intents.pop(name, None)

    def add_entity(self, name, lines):
        with self.compile_lock:
            self.must_compile = True
            self._mutation_gen += 1
            self.entity_lines[name] = lines

    def remove_entity(self, name):
        with self.compile_lock:
            self.must_compile = True
            self._mutation_gen += 1
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

    @staticmethod
    def _referenced_entities(intent_name, lines, entities):
        """The inlined entity patterns ``_create_intent_pattern`` could
        substitute into these lines, as a hashable cache key part.

        A substitution only happens where the line holds the literal
        ``{name}`` (the entity's own name, or its name without this skill's
        namespace), and ``_create_pattern`` only ever adds characters around
        braces, so scanning the raw lines for that text can only over-count:
        an unrelated entity change may invalidate an intent, never the
        reverse.
        """
        namespace = intent_name.split(':')[0] + ':'
        text = '\n'.join(lines)
        referenced = []
        for ent_name, pattern in entities.items():
            slot = ent_name[len(namespace):] if ent_name.startswith(namespace) else ent_name
            if '{' + slot + '}' in text:
                referenced.append((ent_name, pattern))
        # in the entities' iteration order, not sorted: when both "colour"
        # and "skill:colour" fill {colour}, the later one in that order wins
        # the substitution, and the key has to change when the order does
        return tuple(referenced)

    def _create_intent_pattern(self, line, intent_name, entities, counter):
        namespace = intent_name.split(':')[0] + ':'
        line = self._cap_line_alternations(line)
        line = self._patterns.get(line, self._create_pattern)
        replacements = {}
        for ent_name in set(re.findall(r'{([a-z_:]+)}', line)):
            replacements[ent_name] = r'(?P<{}__{{}}>.*?\w.*?)'.format(ent_name)
        for ent_name, ent in entities.items():
            ent_regex = r'(?P<{}__{{}}>{})'
            if ent_name.startswith(namespace):
                replacements[ent_name[len(namespace):]] = ent_regex.format(
                    ent_name[len(namespace):], ent
                )
            else:
                replacements[ent_name] = ent_regex.format(ent_name.replace(':', '__colon__'), ent)
        for key, value in replacements.items():
            line = line.replace('{' + key + '}', value.format(next(counter)), 1)
        return '^{}$'.format(line)

    def _create_regex(self, line, intent_name, entities, counter):
        """ Create regex and return. If error occurs returns None. """
        try:
            return re.compile(
                self._create_intent_pattern(line, intent_name, entities, counter),
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

    def create_regexes(self, lines, intent_name, entities, counter):
        regexes = [self._create_regex(line, intent_name, entities, counter)
                   for line in sorted(lines, key=len, reverse=True)
                   if line.strip() and not _PLAIN_LINE.match(line)]
        # Filter out all regexes that fails
        return [r for r in regexes if r is not None]

    @staticmethod
    def _plain_words(lines):
        """The word tuples of an intent's plain lines (``_PLAIN_LINE``).

        Such a line becomes ``^\\W*w1\\W+...w2\\W*$`` under
        ``_create_pattern`` (symbols optional, matched case-insensitively): a
        query matches it exactly when the query's words, split on non-word
        characters, are the line's words. So such a line needs no regex at all: ``calc_intents``
        compares word tuples, and a match carries no entities, which is also
        the least-greedy answer any line of the intent could give.
        """
        return frozenset(_words(line) for line in lines if _PLAIN_LINE.match(line))

    @staticmethod
    def _with_prefilter(regexes, plain=frozenset()):
        """Wrap an intent's line regexes in an ``_IntentRegexes`` carrying a
        prefilter: one regex that is the alternation of them all, or None
        when the intent has no lines (or just one, which is its own
        prefilter).

        Each line pattern is already ``^body$`` and every capture group
        name is globally unique (the shared counter in
        ``_create_intent_pattern``), so ``(?:^b1$)|(?:^b2$)|...`` compiles
        without a duplicate-group clash and matches iff at least one line
        matches. The prefilter is a membership test only; the caller re-runs
        the per-line scan to pick the winning line.
        """
        # the prefilter itself is built on first use (see _IntentRegexes)
        return _IntentRegexes(regexes, plain)

    def compile(self):
        """Compile the container. Callers must serialize compiles among
        themselves: two overlapping passes each publish their own entity
        set wholesale, so a slower older pass can overwrite a newer one's
        entities and leave the container reporting itself clean.
        ``IntentContainer.train`` - the only caller - holds ``_train_lock``
        for exactly this reason.
        """
        self._compile()

    def _compile(self):
        """Rebuild the compiled state and swap it in.

        The compile itself runs against a private snapshot with no lock
        held: ``add_intent``/``add_entity`` and friends are called from the
        bus thread that also delivers queries, so holding ``compile_lock``
        across a compile that runs for tens of seconds stalls every message
        queued behind the registration waiting on it. Only the snapshot and
        the swap take the lock.
        """
        with self.compile_lock:
            start_gen = self._mutation_gen
            entity_lines = dict(self.entity_lines)
            intent_lines = dict(self.intent_lines)

        start = time.monotonic()
        largest_entity, largest_size = None, 0
        entities, capped_entities = {}, set()
        entity_cache = {}
        for ent_name, lines in entity_lines.items():
            values = [line for line in lines if line.strip()]
            if len(values) > largest_size:
                largest_entity, largest_size = ent_name, len(values)
            if len(values) > PADAOS_ENTITY_INLINE_CAP:
                capped_entities.add(ent_name)
                # too many values to inline: skip so referencing slots fall
                # back to the plain wildcard capture (see PADAOS_ENTITY_INLINE_CAP)
                continue
            key = tuple(values)
            cached = self._entity_cache.get(ent_name)
            if cached is not None and cached[0] == key:
                pattern = cached[1]
            else:
                pattern = r'({})'.format('|'.join(
                    self._patterns.get(line, self._create_pattern) for line in values
                ))
            entity_cache[ent_name] = (key, pattern)
            entities[ent_name] = pattern
        # Every pass used to rebuild every intent of the container, so a
        # runtime whose skills register one after another paid for the whole
        # container again on each of them (quadratic in the skill count, and
        # once per language). An intent whose lines and referenced entities
        # are unchanged keeps the regexes the previous pass built for it.
        intent_cache = {}
        intents = {}
        for intent_name, lines in intent_lines.items():
            key = (tuple(lines),
                   self._referenced_entities(intent_name, lines, entities))
            cached = self._intent_cache.get(intent_name)
            if cached is not None and cached[0] == key:
                compiled = cached[1]
            else:
                # Capture-group names only have to be unique within one
                # regex (each line regex, and the prefilter joining one
                # intent's lines), so each intent numbers its own groups and
                # its regexes do not depend on what else is registered.
                # each intent's regexes travel with their prefilter in one
                # object, both built off-lock like the regexes themselves.
                compiled = self._with_prefilter(
                    self.create_regexes(lines, intent_name, entities, count()),
                    self._plain_words(lines))
            intent_cache[intent_name] = (key, compiled)
            intents[intent_name] = compiled
        self._entity_cache = entity_cache
        self._intent_cache = intent_cache
        self._patterns.save()
        duration = time.monotonic() - start

        with self.compile_lock:
            # A registration that landed while this pass ran already
            # retired its own compiled entry (see add_intent/remove_intent);
            # publishing this pass's result for that name would resurrect
            # the definition it just replaced or removed, so keep only the
            # names whose source lines are still exactly what was compiled.
            self.intents.update({
                name: regexes for name, regexes in intents.items()
                if self.intent_lines.get(name) is intent_lines[name]
            })
            self.entities = entities
            self.capped_entities = capped_entities
            if self._mutation_gen == start_gen:
                self.must_compile = False

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
                capped = largest_entity in capped_entities
                LOG.warning(
                    f"padaos compile took {duration:.2f}s for the full "
                    f"container ({len(entity_lines)} entities, "
                    f"{len(intent_lines)} intents); largest entity "
                    f"'{largest_entity}' has {largest_size} values"
                    f"{' (capped, not inlined)' if capped else ''}"
                )

    @staticmethod
    def _match_entities(regexes, query):
        """Return the least-greedy entity dict among an intent's line
        regexes, or None if none match.

        Least-greedy is the same rule ``min(..., key=sum-of-value-lengths)``
        applied before, kept here to skip allocating a per-intent list of
        every matching line's groupdict (most intents match zero lines, so
        the list and the generator frame were pure overhead on the hot
        query path). A strict ``<`` keeps the first line at a tie, matching
        ``min``'s first-wins behaviour exactly.
        """
        best, best_len = None, 0
        for regex in regexes:
            match = regex.match(query)
            if match is None:
                continue
            groups = {
                k.rsplit('__', 1)[0].replace('__colon__', ':'): v.strip()
                for k, v in match.groupdict().items() if v
            }
            total = 0
            for v in groups.values():
                total += len(v)
            if best is None or total < best_len:
                best, best_len = groups, total
        return best

    def calc_intents(self, query):
        query = ' ' + query + ' '
        if self.must_compile:
            # The match path must NEVER compile: a synchronous compile here
            # runs on whatever thread called calc_intents, which in
            # production is the bus message thread handling the query
            # itself (see IntentContainer.calc_intents ->
            # opm.py's handle_get_padatious). A registration burst that
            # leaves must_compile True (e.g. a hash no-op replay - see
            # IntentContainer.needs_compile) must instead be compiled by
            # the background training worker; here we just serve whatever
            # was compiled last (possibly empty, on a container that has
            # never compiled at all) rather than block the caller.
            if not self._warned_pending_compile:
                LOG.info("padaos compiling in background, "
                         "serving last compiled state in the meantime")
                self._warned_pending_compile = True
        else:
            self._warned_pending_compile = False
        words = _words(query)
        for intent_name, regexes in self.intents.items():
            if words in regexes.plain:
                # a plain line matched: no entities, which no other line of
                # the intent can beat (see _plain_words)
                yield {'name': intent_name, 'entities': {}}
                continue
            prefilter = regexes.built_prefilter
            if prefilter is not None and prefilter.match(query) is None:
                # one C-level match rules the whole intent out; the vast
                # majority of intents are rejected here without touching the
                # per-line Python loop. A missing prefilter never skips.
                continue
            entities = self._match_entities(regexes, query)
            if entities is not None:
                yield {'name': intent_name, 'entities': entities}

    def calc_intent(self, query):
        return min(
            self.calc_intents(query),
            key=lambda x: sum(map(len, x['entities'].values())),
            default={'name': None, 'entities': {}}
        )
