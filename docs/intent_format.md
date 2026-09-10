# Intent & Entity File Format

Padatious intents are plain text files where each line is a training sample.
The syntax supports optional words, alternatives, and named entity slots.

## Syntax Elements

### Plain words

Any sequence of words is matched literally (case-insensitive).

```
turn on the lights
play some music
```

### Alternatives `(a|b|c)`

Parentheses with pipe-separated options expand to multiple training samples.

```
(turn on|switch on|enable) the lights
play (some|a bit of) music
```

An empty alternative makes the group optional:

```
(please |)turn on the lights   # "please" is optional
```

### Entity slots `{name}`

Curly-brace references are named capture groups. The extracted value is returned in `MatchData.matches`.

```
search for {query} on {engine}
remind me to {task} at {time}
```

Entity names may only contain lowercase letters and underscores.

### Combining alternatives and slots

```
(search|look) for {query} (on|using|with) {engine}
set a {duration} (timer|alarm)
```

### Special tokens

| Token | Meaning |
|---|---|
| `#` | Matches any digit (`\d`) |
| `:0` | Wildcard, matches one or more words |

```
call extension ##      # matches "call extension 42"
play :0                # matches "play anything here"
```

## Intent files (`.intent`)

Each line is one sample utterance. Blank lines and leading/trailing whitespace are ignored.

```
# home.lights.intent
turn on the (living room |bedroom |)lights
switch on (all |)the lights
lights on
```

Load a file into a container:

```python
container.load_intent('home.lights', 'home.lights.intent')
```

Or pass samples inline:

```python
container.add_intent('home.lights', [
    'turn on the (living room |bedroom |)lights',
    'switch on (all |)the lights',
    'lights on',
])
```

## Entity files (`.entity`)

Each line is one example value for the entity. These constrain what the neural network considers a valid match for the slot.

```
# weekday.entity
monday
tuesday
wednesday
thursday
friday
```

Load and use:

```python
container.add_intent('weather', ['will it rain on {weekday}?'])
container.load_entity('weekday', 'weekday.entity')
container.train()
```

Without an entity file the slot will still capture any text, but the neural network match may be less precise.

## Blacklisted words

When registering an intent you can pass `blacklisted_words` to suppress matches that contain specific words:

```python
container.add_intent('play_music', ['play {song}'], blacklisted_words=['video'])
```

The intent will not fire if the word `"video"` appears anywhere in the utterance.

## Tips

- More varied training samples generally improve accuracy.
- Keep entity files focused — only include realistic values.
- Use alternatives to cover common phrasings rather than writing each one separately.
- Intents are independent. The engine scores all of them and returns the best match.

---
[← Installation](installation.md) · [Home](README.md) · [API Reference →](api_reference.md)
