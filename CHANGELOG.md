# Changelog

## [2.0.16a1](https://github.com/OpenVoiceOS/ovos-padatious-pipeline-plugin/tree/2.0.16a1) (2026-09-02)

[Full Changelog](https://github.com/OpenVoiceOS/ovos-padatious-pipeline-plugin/compare/2.0.15a1...2.0.16a1)

**Merged pull requests:**

- fix: an intent is matchable as soon as it is registered [\#128](https://github.com/OpenVoiceOS/ovos-padatious-pipeline-plugin/pull/128) ([JarbasAl](https://github.com/JarbasAl))

## [2.0.15a1](https://github.com/OpenVoiceOS/ovos-padatious-pipeline-plugin/tree/2.0.15a1) (2026-09-01)

[Full Changelog](https://github.com/OpenVoiceOS/ovos-padatious-pipeline-plugin/compare/2.0.14a1...2.0.15a1)

**Merged pull requests:**

- fix: matching never waits on an in-flight compile [\#126](https://github.com/OpenVoiceOS/ovos-padatious-pipeline-plugin/pull/126) ([JarbasAl](https://github.com/JarbasAl))

## [2.0.14a1](https://github.com/OpenVoiceOS/ovos-padatious-pipeline-plugin/tree/2.0.14a1) (2026-09-01)

[Full Changelog](https://github.com/OpenVoiceOS/ovos-padatious-pipeline-plugin/compare/2.0.13a1...2.0.14a1)

**Merged pull requests:**

- fix: padaos compiles in the background, never on the match path [\#124](https://github.com/OpenVoiceOS/ovos-padatious-pipeline-plugin/pull/124) ([JarbasAl](https://github.com/JarbasAl))

## [2.0.13a1](https://github.com/OpenVoiceOS/ovos-padatious-pipeline-plugin/tree/2.0.13a1) (2026-09-01)

[Full Changelog](https://github.com/OpenVoiceOS/ovos-padatious-pipeline-plugin/compare/2.0.12a1...2.0.13a1)

**Merged pull requests:**

- fix: identical re-registration is a no-op and retrains never blank a live intent [\#122](https://github.com/OpenVoiceOS/ovos-padatious-pipeline-plugin/pull/122) ([JarbasAl](https://github.com/JarbasAl))

## [2.0.12a1](https://github.com/OpenVoiceOS/ovos-padatious-pipeline-plugin/tree/2.0.12a1) (2026-09-01)

[Full Changelog](https://github.com/OpenVoiceOS/ovos-padatious-pipeline-plugin/compare/2.0.11a1...2.0.12a1)

**Merged pull requests:**

- fix: cap literal intent-line alternation groups [\#120](https://github.com/OpenVoiceOS/ovos-padatious-pipeline-plugin/pull/120) ([JarbasAl](https://github.com/JarbasAl))

## [2.0.11a1](https://github.com/OpenVoiceOS/ovos-padatious-pipeline-plugin/tree/2.0.11a1) (2026-09-01)

[Full Changelog](https://github.com/OpenVoiceOS/ovos-padatious-pipeline-plugin/compare/2.0.10a1...2.0.11a1)

**Merged pull requests:**

- fix: background training publishes complete state and never blocks the bus getters [\#118](https://github.com/OpenVoiceOS/ovos-padatious-pipeline-plugin/pull/118) ([JarbasAl](https://github.com/JarbasAl))

## [2.0.10a1](https://github.com/OpenVoiceOS/ovos-padatious-pipeline-plugin/tree/2.0.10a1) (2026-08-31)

[Full Changelog](https://github.com/OpenVoiceOS/ovos-padatious-pipeline-plugin/compare/2.0.9a1...2.0.10a1)

**Merged pull requests:**

- fix: bound padaos entity alternations and train off the utterance thread [\#115](https://github.com/OpenVoiceOS/ovos-padatious-pipeline-plugin/pull/115) ([JarbasAl](https://github.com/JarbasAl))

## [2.0.9a1](https://github.com/OpenVoiceOS/ovos-padatious-pipeline-plugin/tree/2.0.9a1) (2026-08-31)

[Full Changelog](https://github.com/OpenVoiceOS/ovos-padatious-pipeline-plugin/compare/2.0.8a2...2.0.9a1)

## [2.0.8a2](https://github.com/OpenVoiceOS/ovos-padatious-pipeline-plugin/tree/2.0.8a2) (2026-08-31)

[Full Changelog](https://github.com/OpenVoiceOS/ovos-padatious-pipeline-plugin/compare/2.0.8a1...2.0.8a2)

**Merged pull requests:**

- fix: deprecate the inline \# digit wildcard in templates [\#111](https://github.com/OpenVoiceOS/ovos-padatious-pipeline-plugin/pull/111) ([JarbasAl](https://github.com/JarbasAl))

## [2.0.8a1](https://github.com/OpenVoiceOS/ovos-padatious-pipeline-plugin/tree/2.0.8a1) (2026-08-31)

[Full Changelog](https://github.com/OpenVoiceOS/ovos-padatious-pipeline-plugin/compare/2.0.7a1...2.0.8a1)

**Merged pull requests:**

- docs: add AGENTS.md with the conventions for coding agents [\#110](https://github.com/OpenVoiceOS/ovos-padatious-pipeline-plugin/pull/110) ([JarbasAl](https://github.com/JarbasAl))
- fix: tolerate malformed templates during intent registration [\#94](https://github.com/OpenVoiceOS/ovos-padatious-pipeline-plugin/pull/94) ([JarbasAl](https://github.com/JarbasAl))

## [2.0.7a1](https://github.com/OpenVoiceOS/ovos-padatious-pipeline-plugin/tree/2.0.7a1) (2026-08-15)

[Full Changelog](https://github.com/OpenVoiceOS/ovos-padatious-pipeline-plugin/compare/2.0.6a1...2.0.7a1)

**Merged pull requests:**

- fix: intent suppression covers exact matches and uses word boundaries [\#108](https://github.com/OpenVoiceOS/ovos-padatious-pipeline-plugin/pull/108) ([JarbasAl](https://github.com/JarbasAl))

## [2.0.6a1](https://github.com/OpenVoiceOS/ovos-padatious-pipeline-plugin/tree/2.0.6a1) (2026-08-15)

[Full Changelog](https://github.com/OpenVoiceOS/ovos-padatious-pipeline-plugin/compare/2.0.5a1...2.0.6a1)

**Merged pull requests:**

- fix: slot blacklists match by whole-value equality [\#106](https://github.com/OpenVoiceOS/ovos-padatious-pipeline-plugin/pull/106) ([JarbasAl](https://github.com/JarbasAl))

## [2.0.5a1](https://github.com/OpenVoiceOS/ovos-padatious-pipeline-plugin/tree/2.0.5a1) (2026-08-14)

[Full Changelog](https://github.com/OpenVoiceOS/ovos-padatious-pipeline-plugin/compare/2.0.4a2...2.0.5a1)

**Merged pull requests:**

- fix: bound entity net training and score listed values exactly [\#103](https://github.com/OpenVoiceOS/ovos-padatious-pipeline-plugin/pull/103) ([JarbasAl](https://github.com/JarbasAl))

## [2.0.4a2](https://github.com/OpenVoiceOS/ovos-padatious-pipeline-plugin/tree/2.0.4a2) (2026-08-14)

[Full Changelog](https://github.com/OpenVoiceOS/ovos-padatious-pipeline-plugin/compare/2.0.4a1...2.0.4a2)

**Merged pull requests:**

- docs: add prerelease-quirks changelog since 1.4.3 [\#101](https://github.com/OpenVoiceOS/ovos-padatious-pipeline-plugin/pull/101) ([JarbasAl](https://github.com/JarbasAl))

## [2.0.4a1](https://github.com/OpenVoiceOS/ovos-padatious-pipeline-plugin/tree/2.0.4a1) (2026-08-14)

[Full Changelog](https://github.com/OpenVoiceOS/ovos-padatious-pipeline-plugin/compare/2.0.3a1...2.0.4a1)

**Merged pull requests:**

- fix: tokenize\(\) splits underscore/digit slot names, breaking §5.4 hint guarantee [\#99](https://github.com/OpenVoiceOS/ovos-padatious-pipeline-plugin/pull/99) ([JarbasAl](https://github.com/JarbasAl))

## [2.0.3a1](https://github.com/OpenVoiceOS/ovos-padatious-pipeline-plugin/tree/2.0.3a1) (2026-08-13)

[Full Changelog](https://github.com/OpenVoiceOS/ovos-padatious-pipeline-plugin/compare/2.0.2a1...2.0.3a1)

**Merged pull requests:**

- fix: entity value sets bias confidence instead of closing the vocabulary [\#97](https://github.com/OpenVoiceOS/ovos-padatious-pipeline-plugin/pull/97) ([JarbasAl](https://github.com/JarbasAl))

## [2.0.2a1](https://github.com/OpenVoiceOS/ovos-padatious-pipeline-plugin/tree/2.0.2a1) (2026-08-13)

[Full Changelog](https://github.com/OpenVoiceOS/ovos-padatious-pipeline-plugin/compare/2.0.1a2...2.0.2a1)

**Merged pull requests:**

- fix: collapse munged entity names so file-registered slots actually constrain [\#95](https://github.com/OpenVoiceOS/ovos-padatious-pipeline-plugin/pull/95) ([JarbasAl](https://github.com/JarbasAl))

## [2.0.1a2](https://github.com/OpenVoiceOS/ovos-padatious-pipeline-plugin/tree/2.0.1a2) (2026-07-31)

[Full Changelog](https://github.com/OpenVoiceOS/ovos-padatious-pipeline-plugin/compare/2.0.1a1...2.0.1a2)

**Merged pull requests:**

- docs: rewrite README in Simplified Technical English [\#91](https://github.com/OpenVoiceOS/ovos-padatious-pipeline-plugin/pull/91) ([JarbasAl](https://github.com/JarbasAl))

## [2.0.1a1](https://github.com/OpenVoiceOS/ovos-padatious-pipeline-plugin/tree/2.0.1a1) (2026-07-26)

[Full Changelog](https://github.com/OpenVoiceOS/ovos-padatious-pipeline-plugin/compare/2.0.0a1...2.0.1a1)

**Merged pull requests:**

- fix: session blacklist bypassed by the legacy/INTENT-4 intent-name alias [\#89](https://github.com/OpenVoiceOS/ovos-padatious-pipeline-plugin/pull/89) ([JarbasAl](https://github.com/JarbasAl))

## [2.0.0a1](https://github.com/OpenVoiceOS/ovos-padatious-pipeline-plugin/tree/2.0.0a1) (2026-07-17)

[Full Changelog](https://github.com/OpenVoiceOS/ovos-padatious-pipeline-plugin/compare/1.9.1a1...2.0.0a1)

**Breaking changes:**

- feat!: pure numpy neural network backend, drop fann2 dependency [\#87](https://github.com/OpenVoiceOS/ovos-padatious-pipeline-plugin/pull/87) ([JarbasAl](https://github.com/JarbasAl))

## [1.9.1a1](https://github.com/OpenVoiceOS/ovos-padatious-pipeline-plugin/tree/1.9.1a1) (2026-07-04)

[Full Changelog](https://github.com/OpenVoiceOS/ovos-padatious-pipeline-plugin/compare/1.9.0a1...1.9.1a1)

**Merged pull requests:**

- chore: typing and docstrs [\#21](https://github.com/OpenVoiceOS/ovos-padatious-pipeline-plugin/pull/21) ([JarbasAl](https://github.com/JarbasAl))

## [1.9.0a1](https://github.com/OpenVoiceOS/ovos-padatious-pipeline-plugin/tree/1.9.0a1) (2026-07-03)

[Full Changelog](https://github.com/OpenVoiceOS/ovos-padatious-pipeline-plugin/compare/1.8.1a1...1.9.0a1)

**Merged pull requests:**

- feat: fill unresolved template slots from context \(OVOS-CONTEXT-1 §7\) [\#82](https://github.com/OpenVoiceOS/ovos-padatious-pipeline-plugin/pull/82) ([JarbasAl](https://github.com/JarbasAl))

## [1.8.1a1](https://github.com/OpenVoiceOS/ovos-padatious-pipeline-plugin/tree/1.8.1a1) (2026-07-03)

[Full Changelog](https://github.com/OpenVoiceOS/ovos-padatious-pipeline-plugin/compare/1.8.0a1...1.8.1a1)

**Merged pull requests:**

- fix: forward blacklisted\_words to add\_intent by keyword [\#83](https://github.com/OpenVoiceOS/ovos-padatious-pipeline-plugin/pull/83) ([JarbasAl](https://github.com/JarbasAl))

## [1.8.0a1](https://github.com/OpenVoiceOS/ovos-padatious-pipeline-plugin/tree/1.8.0a1) (2026-07-02)

[Full Changelog](https://github.com/OpenVoiceOS/ovos-padatious-pipeline-plugin/compare/1.7.1a1...1.8.0a1)

**Merged pull requests:**

- feat: enforce OVOS-CONTEXT-1 requires/excludes\_context gating at match time [\#80](https://github.com/OpenVoiceOS/ovos-padatious-pipeline-plugin/pull/80) ([JarbasAl](https://github.com/JarbasAl))

## [1.7.1a1](https://github.com/OpenVoiceOS/ovos-padatious-pipeline-plugin/tree/1.7.1a1) (2026-06-28)

[Full Changelog](https://github.com/OpenVoiceOS/ovos-padatious-pipeline-plugin/compare/1.7.0a1...1.7.1a1)

**Merged pull requests:**

- fix: lift ovos-spec-tools upper bound \(spec-tools 1.x\) [\#78](https://github.com/OpenVoiceOS/ovos-padatious-pipeline-plugin/pull/78) ([JarbasAl](https://github.com/JarbasAl))

## [1.7.0a1](https://github.com/OpenVoiceOS/ovos-padatious-pipeline-plugin/tree/1.7.0a1) (2026-06-28)

[Full Changelog](https://github.com/OpenVoiceOS/ovos-padatious-pipeline-plugin/compare/1.6.2a1...1.7.0a1)

**Merged pull requests:**

- feat: consume OVOS-INTENT-4 template registration \(alongside legacy\) [\#72](https://github.com/OpenVoiceOS/ovos-padatious-pipeline-plugin/pull/72) ([JarbasAl](https://github.com/JarbasAl))

## [1.6.2a1](https://github.com/OpenVoiceOS/ovos-padatious-pipeline-plugin/tree/1.6.2a1) (2026-06-27)

[Full Changelog](https://github.com/OpenVoiceOS/ovos-padatious-pipeline-plugin/compare/1.6.1a1...1.6.2a1)

**Merged pull requests:**

- fix: drop unhashable Session from lru\_cache key \(ovos-bus-client 2.x compat\) [\#75](https://github.com/OpenVoiceOS/ovos-padatious-pipeline-plugin/pull/75) ([JarbasAl](https://github.com/JarbasAl))

## [1.6.1a1](https://github.com/OpenVoiceOS/ovos-padatious-pipeline-plugin/tree/1.6.1a1) (2026-06-27)

[Full Changelog](https://github.com/OpenVoiceOS/ovos-padatious-pipeline-plugin/compare/1.6.0a1...1.6.1a1)

**Merged pull requests:**

- fix\(deps\): allow ovos-workshop 9.x \(widen \<9.0.0 -\> \<10.0.0\) [\#73](https://github.com/OpenVoiceOS/ovos-padatious-pipeline-plugin/pull/73) ([JarbasAl](https://github.com/JarbasAl))

## [1.6.0a1](https://github.com/OpenVoiceOS/ovos-padatious-pipeline-plugin/tree/1.6.0a1) (2026-06-24)

[Full Changelog](https://github.com/OpenVoiceOS/ovos-padatious-pipeline-plugin/compare/1.5.0a1...1.6.0a1)

**Merged pull requests:**

- refactor: migrate to ovos-spec-tools [\#70](https://github.com/OpenVoiceOS/ovos-padatious-pipeline-plugin/pull/70) ([JarbasAl](https://github.com/JarbasAl))

## [1.5.0a1](https://github.com/OpenVoiceOS/ovos-padatious-pipeline-plugin/tree/1.5.0a1) (2026-05-14)

[Full Changelog](https://github.com/OpenVoiceOS/ovos-padatious-pipeline-plugin/compare/1.4.5a2...1.5.0a1)

**Merged pull requests:**

- fix + test: clear stale intent cache on train/detach; add ovoscope e2e suite [\#67](https://github.com/OpenVoiceOS/ovos-padatious-pipeline-plugin/pull/67) ([JarbasAl](https://github.com/JarbasAl))

## [1.4.5a2](https://github.com/OpenVoiceOS/ovos-padatious-pipeline-plugin/tree/1.4.5a2) (2026-04-09)

[Full Changelog](https://github.com/OpenVoiceOS/ovos-padatious-pipeline-plugin/compare/1.4.5a1...1.4.5a2)

**Merged pull requests:**

- chore\(ovos-padatious\): allow ovos-workshop\<9.0.0 [\#65](https://github.com/OpenVoiceOS/ovos-padatious-pipeline-plugin/pull/65) ([JarbasAl](https://github.com/JarbasAl))

## [1.4.5a1](https://github.com/OpenVoiceOS/ovos-padatious-pipeline-plugin/tree/1.4.5a1) (2026-03-03)

[Full Changelog](https://github.com/OpenVoiceOS/ovos-padatious-pipeline-plugin/compare/1.4.4a2...1.4.5a1)

**Merged pull requests:**

- fix: default config values [\#61](https://github.com/OpenVoiceOS/ovos-padatious-pipeline-plugin/pull/61) ([JarbasAl](https://github.com/JarbasAl))

## [1.4.4a2](https://github.com/OpenVoiceOS/ovos-padatious-pipeline-plugin/tree/1.4.4a2) (2026-03-03)

[Full Changelog](https://github.com/OpenVoiceOS/ovos-padatious-pipeline-plugin/compare/1.4.4a1...1.4.4a2)

**Merged pull requests:**

- Docs: Add comprehensive documentation including API reference, architecture, and theory [\#62](https://github.com/OpenVoiceOS/ovos-padatious-pipeline-plugin/pull/62) ([JarbasAl](https://github.com/JarbasAl))

## [1.4.4a1](https://github.com/OpenVoiceOS/ovos-padatious-pipeline-plugin/tree/1.4.4a1) (2025-12-16)

[Full Changelog](https://github.com/OpenVoiceOS/ovos-padatious-pipeline-plugin/compare/1.4.3...1.4.4a1)

**Merged pull requests:**

- fix: thread safety [\#59](https://github.com/OpenVoiceOS/ovos-padatious-pipeline-plugin/pull/59) ([JarbasAl](https://github.com/JarbasAl))

## [1.4.3](https://github.com/OpenVoiceOS/ovos-padatious-pipeline-plugin/tree/1.4.3) (2025-11-05)

[Full Changelog](https://github.com/OpenVoiceOS/ovos-padatious-pipeline-plugin/compare/1.4.3a1...1.4.3)

**Merged pull requests:**

- Release 1.4.3a1 [\#58](https://github.com/OpenVoiceOS/ovos-padatious-pipeline-plugin/pull/58) ([github-actions[bot]](https://github.com/apps/github-actions))

## [1.4.3a1](https://github.com/OpenVoiceOS/ovos-padatious-pipeline-plugin/tree/1.4.3a1) (2025-11-05)

[Full Changelog](https://github.com/OpenVoiceOS/ovos-padatious-pipeline-plugin/compare/1.4.2...1.4.3a1)

**Merged pull requests:**

- Update requirements.txt [\#57](https://github.com/OpenVoiceOS/ovos-padatious-pipeline-plugin/pull/57) ([JarbasAl](https://github.com/JarbasAl))

## [1.4.2](https://github.com/OpenVoiceOS/ovos-padatious-pipeline-plugin/tree/1.4.2) (2025-06-08)

[Full Changelog](https://github.com/OpenVoiceOS/ovos-padatious-pipeline-plugin/compare/1.4.2a1...1.4.2)

**Merged pull requests:**

- Release 1.4.2a1 [\#56](https://github.com/OpenVoiceOS/ovos-padatious-pipeline-plugin/pull/56) ([github-actions[bot]](https://github.com/apps/github-actions))

## [1.4.2a1](https://github.com/OpenVoiceOS/ovos-padatious-pipeline-plugin/tree/1.4.2a1) (2025-06-08)

[Full Changelog](https://github.com/OpenVoiceOS/ovos-padatious-pipeline-plugin/compare/1.4.1...1.4.2a1)

**Merged pull requests:**

- fix:  deprecated\_code\_import\_error [\#55](https://github.com/OpenVoiceOS/ovos-padatious-pipeline-plugin/pull/55) ([JarbasAl](https://github.com/JarbasAl))

## [1.4.1](https://github.com/OpenVoiceOS/ovos-padatious-pipeline-plugin/tree/1.4.1) (2025-06-08)

[Full Changelog](https://github.com/OpenVoiceOS/ovos-padatious-pipeline-plugin/compare/1.4.1a1...1.4.1)

**Merged pull requests:**

- Release 1.4.1a1 [\#54](https://github.com/OpenVoiceOS/ovos-padatious-pipeline-plugin/pull/54) ([github-actions[bot]](https://github.com/apps/github-actions))

## [1.4.1a1](https://github.com/OpenVoiceOS/ovos-padatious-pipeline-plugin/tree/1.4.1a1) (2025-06-08)

[Full Changelog](https://github.com/OpenVoiceOS/ovos-padatious-pipeline-plugin/compare/1.4.0a1...1.4.1a1)

**Merged pull requests:**

- fix: compatibility with ovos-workshop 7.X.X and ovos-plugin-manager 1.X.X [\#53](https://github.com/OpenVoiceOS/ovos-padatious-pipeline-plugin/pull/53) ([JarbasAl](https://github.com/JarbasAl))

## [1.4.0a1](https://github.com/OpenVoiceOS/ovos-padatious-pipeline-plugin/tree/1.4.0a1) (2025-04-03)

[Full Changelog](https://github.com/OpenVoiceOS/ovos-padatious-pipeline-plugin/compare/1.3.6...1.4.0a1)

**Merged pull requests:**

- Release 1.4.0a1 [\#52](https://github.com/OpenVoiceOS/ovos-padatious-pipeline-plugin/pull/52) ([github-actions[bot]](https://github.com/apps/github-actions))
- feat:blacklisted words [\#51](https://github.com/OpenVoiceOS/ovos-padatious-pipeline-plugin/pull/51) ([JarbasAl](https://github.com/JarbasAl))

## [1.3.6](https://github.com/OpenVoiceOS/ovos-padatious-pipeline-plugin/tree/1.3.6) (2025-02-27)

[Full Changelog](https://github.com/OpenVoiceOS/ovos-padatious-pipeline-plugin/compare/1.3.6a1...1.3.6)

**Merged pull requests:**

- Release 1.3.6a1 [\#50](https://github.com/OpenVoiceOS/ovos-padatious-pipeline-plugin/pull/50) ([github-actions[bot]](https://github.com/apps/github-actions))

## [1.3.6a1](https://github.com/OpenVoiceOS/ovos-padatious-pipeline-plugin/tree/1.3.6a1) (2025-02-27)

[Full Changelog](https://github.com/OpenVoiceOS/ovos-padatious-pipeline-plugin/compare/1.3.5a1...1.3.6a1)

**Merged pull requests:**

- fix: thread safety, avoid some types of fann2 crashes [\#49](https://github.com/OpenVoiceOS/ovos-padatious-pipeline-plugin/pull/49) ([JarbasAl](https://github.com/JarbasAl))

## [1.3.5a1](https://github.com/OpenVoiceOS/ovos-padatious-pipeline-plugin/tree/1.3.5a1) (2025-02-02)

[Full Changelog](https://github.com/OpenVoiceOS/ovos-padatious-pipeline-plugin/compare/1.3.4...1.3.5a1)

**Merged pull requests:**

- Release 1.3.5a1 [\#48](https://github.com/OpenVoiceOS/ovos-padatious-pipeline-plugin/pull/48) ([github-actions[bot]](https://github.com/apps/github-actions))
- refactor:shared utils [\#47](https://github.com/OpenVoiceOS/ovos-padatious-pipeline-plugin/pull/47) ([JarbasAl](https://github.com/JarbasAl))

## [1.3.4](https://github.com/OpenVoiceOS/ovos-padatious-pipeline-plugin/tree/1.3.4) (2025-01-29)

[Full Changelog](https://github.com/OpenVoiceOS/ovos-padatious-pipeline-plugin/compare/1.3.4a1...1.3.4)

**Merged pull requests:**

- Release 1.3.4a1 [\#45](https://github.com/OpenVoiceOS/ovos-padatious-pipeline-plugin/pull/45) ([github-actions[bot]](https://github.com/apps/github-actions))

## [1.3.4a1](https://github.com/OpenVoiceOS/ovos-padatious-pipeline-plugin/tree/1.3.4a1) (2025-01-29)

[Full Changelog](https://github.com/OpenVoiceOS/ovos-padatious-pipeline-plugin/compare/1.3.3...1.3.4a1)

**Merged pull requests:**

- fix:domain engine from config [\#44](https://github.com/OpenVoiceOS/ovos-padatious-pipeline-plugin/pull/44) ([JarbasAl](https://github.com/JarbasAl))

## [1.3.3](https://github.com/OpenVoiceOS/ovos-padatious-pipeline-plugin/tree/1.3.3) (2025-01-26)

[Full Changelog](https://github.com/OpenVoiceOS/ovos-padatious-pipeline-plugin/compare/1.3.3a1...1.3.3)

**Merged pull requests:**

- Release 1.3.3a1 [\#43](https://github.com/OpenVoiceOS/ovos-padatious-pipeline-plugin/pull/43) ([github-actions[bot]](https://github.com/apps/github-actions))

## [1.3.3a1](https://github.com/OpenVoiceOS/ovos-padatious-pipeline-plugin/tree/1.3.3a1) (2025-01-26)

[Full Changelog](https://github.com/OpenVoiceOS/ovos-padatious-pipeline-plugin/compare/1.3.2...1.3.3a1)

**Merged pull requests:**

- Enhance Padatious intent management with state restoration and training improvements [\#42](https://github.com/OpenVoiceOS/ovos-padatious-pipeline-plugin/pull/42) ([JarbasAl](https://github.com/JarbasAl))

## [1.3.2](https://github.com/OpenVoiceOS/ovos-padatious-pipeline-plugin/tree/1.3.2) (2025-01-25)

[Full Changelog](https://github.com/OpenVoiceOS/ovos-padatious-pipeline-plugin/compare/1.3.2a1...1.3.2)

**Merged pull requests:**

- Release 1.3.2a1 [\#41](https://github.com/OpenVoiceOS/ovos-padatious-pipeline-plugin/pull/41) ([github-actions[bot]](https://github.com/apps/github-actions))

## [1.3.2a1](https://github.com/OpenVoiceOS/ovos-padatious-pipeline-plugin/tree/1.3.2a1) (2025-01-25)

[Full Changelog](https://github.com/OpenVoiceOS/ovos-padatious-pipeline-plugin/compare/1.3.1...1.3.2a1)

**Merged pull requests:**

- fix:improve cache [\#40](https://github.com/OpenVoiceOS/ovos-padatious-pipeline-plugin/pull/40) ([JarbasAl](https://github.com/JarbasAl))

## [1.3.1](https://github.com/OpenVoiceOS/ovos-padatious-pipeline-plugin/tree/1.3.1) (2025-01-25)

[Full Changelog](https://github.com/OpenVoiceOS/ovos-padatious-pipeline-plugin/compare/1.3.1a1...1.3.1)

**Merged pull requests:**

- Release 1.3.1a1 [\#39](https://github.com/OpenVoiceOS/ovos-padatious-pipeline-plugin/pull/39) ([github-actions[bot]](https://github.com/apps/github-actions))

## [1.3.1a1](https://github.com/OpenVoiceOS/ovos-padatious-pipeline-plugin/tree/1.3.1a1) (2025-01-25)

[Full Changelog](https://github.com/OpenVoiceOS/ovos-padatious-pipeline-plugin/compare/1.3.0...1.3.1a1)

**Merged pull requests:**

- fix: normalization + disable new feats by default [\#38](https://github.com/OpenVoiceOS/ovos-padatious-pipeline-plugin/pull/38) ([JarbasAl](https://github.com/JarbasAl))

## [1.3.0](https://github.com/OpenVoiceOS/ovos-padatious-pipeline-plugin/tree/1.3.0) (2025-01-24)

[Full Changelog](https://github.com/OpenVoiceOS/ovos-padatious-pipeline-plugin/compare/1.3.0a1...1.3.0)

**Merged pull requests:**

- Release 1.3.0a1 [\#37](https://github.com/OpenVoiceOS/ovos-padatious-pipeline-plugin/pull/37) ([github-actions[bot]](https://github.com/apps/github-actions))

## [1.3.0a1](https://github.com/OpenVoiceOS/ovos-padatious-pipeline-plugin/tree/1.3.0a1) (2025-01-24)

[Full Changelog](https://github.com/OpenVoiceOS/ovos-padatious-pipeline-plugin/compare/1.2.0...1.3.0a1)

**Merged pull requests:**

- feat/domain\_engine [\#31](https://github.com/OpenVoiceOS/ovos-padatious-pipeline-plugin/pull/31) ([JarbasAl](https://github.com/JarbasAl))

## [1.2.0](https://github.com/OpenVoiceOS/ovos-padatious-pipeline-plugin/tree/1.2.0) (2025-01-24)

[Full Changelog](https://github.com/OpenVoiceOS/ovos-padatious-pipeline-plugin/compare/1.2.0a1...1.2.0)

**Merged pull requests:**

- Release 1.2.0a1 [\#36](https://github.com/OpenVoiceOS/ovos-padatious-pipeline-plugin/pull/36) ([github-actions[bot]](https://github.com/apps/github-actions))

## [1.2.0a1](https://github.com/OpenVoiceOS/ovos-padatious-pipeline-plugin/tree/1.2.0a1) (2025-01-24)

[Full Changelog](https://github.com/OpenVoiceOS/ovos-padatious-pipeline-plugin/compare/1.1.1...1.2.0a1)

**Merged pull requests:**

- feat: stemmer [\#35](https://github.com/OpenVoiceOS/ovos-padatious-pipeline-plugin/pull/35) ([JarbasAl](https://github.com/JarbasAl))

## [1.1.1](https://github.com/OpenVoiceOS/ovos-padatious-pipeline-plugin/tree/1.1.1) (2024-12-12)

[Full Changelog](https://github.com/OpenVoiceOS/ovos-padatious-pipeline-plugin/compare/1.1.1a1...1.1.1)

**Merged pull requests:**

- Release 1.1.1a1 [\#33](https://github.com/OpenVoiceOS/ovos-padatious-pipeline-plugin/pull/33) ([github-actions[bot]](https://github.com/apps/github-actions))

## [1.1.1a1](https://github.com/OpenVoiceOS/ovos-padatious-pipeline-plugin/tree/1.1.1a1) (2024-12-11)

[Full Changelog](https://github.com/OpenVoiceOS/ovos-padatious-pipeline-plugin/compare/1.1.0...1.1.1a1)

**Merged pull requests:**

- performance: paralelize inference [\#32](https://github.com/OpenVoiceOS/ovos-padatious-pipeline-plugin/pull/32) ([JarbasAl](https://github.com/JarbasAl))

## [1.1.0](https://github.com/OpenVoiceOS/ovos-padatious-pipeline-plugin/tree/1.1.0) (2024-12-09)

[Full Changelog](https://github.com/OpenVoiceOS/ovos-padatious-pipeline-plugin/compare/1.1.0a1...1.1.0)

**Merged pull requests:**

- Release 1.1.0a1 [\#30](https://github.com/OpenVoiceOS/ovos-padatious-pipeline-plugin/pull/30) ([github-actions[bot]](https://github.com/apps/github-actions))

## [1.1.0a1](https://github.com/OpenVoiceOS/ovos-padatious-pipeline-plugin/tree/1.1.0a1) (2024-12-09)

[Full Changelog](https://github.com/OpenVoiceOS/ovos-padatious-pipeline-plugin/compare/1.0.5...1.1.0a1)

**Merged pull requests:**

- feat:delayed\_padatious\_training [\#29](https://github.com/OpenVoiceOS/ovos-padatious-pipeline-plugin/pull/29) ([JarbasAl](https://github.com/JarbasAl))

## [1.0.5](https://github.com/OpenVoiceOS/ovos-padatious-pipeline-plugin/tree/1.0.5) (2024-12-06)

[Full Changelog](https://github.com/OpenVoiceOS/ovos-padatious-pipeline-plugin/compare/1.0.5a1...1.0.5)

**Merged pull requests:**

- Release 1.0.5a1 [\#27](https://github.com/OpenVoiceOS/ovos-padatious-pipeline-plugin/pull/27) ([github-actions[bot]](https://github.com/apps/github-actions))

## [1.0.5a1](https://github.com/OpenVoiceOS/ovos-padatious-pipeline-plugin/tree/1.0.5a1) (2024-12-06)

[Full Changelog](https://github.com/OpenVoiceOS/ovos-padatious-pipeline-plugin/compare/1.0.4...1.0.5a1)

**Merged pull requests:**

- fix:expand [\#26](https://github.com/OpenVoiceOS/ovos-padatious-pipeline-plugin/pull/26) ([JarbasAl](https://github.com/JarbasAl))

## [1.0.4](https://github.com/OpenVoiceOS/ovos-padatious-pipeline-plugin/tree/1.0.4) (2024-11-19)

[Full Changelog](https://github.com/OpenVoiceOS/ovos-padatious-pipeline-plugin/compare/1.0.4a1...1.0.4)

**Merged pull requests:**

- Release 1.0.4a1 [\#25](https://github.com/OpenVoiceOS/ovos-padatious-pipeline-plugin/pull/25) ([github-actions[bot]](https://github.com/apps/github-actions))

## [1.0.4a1](https://github.com/OpenVoiceOS/ovos-padatious-pipeline-plugin/tree/1.0.4a1) (2024-11-19)

[Full Changelog](https://github.com/OpenVoiceOS/ovos-padatious-pipeline-plugin/compare/1.0.3...1.0.4a1)

**Merged pull requests:**

- fix:allow\_workshop\_300 [\#24](https://github.com/OpenVoiceOS/ovos-padatious-pipeline-plugin/pull/24) ([JarbasAl](https://github.com/JarbasAl))

## [1.0.3](https://github.com/OpenVoiceOS/ovos-padatious-pipeline-plugin/tree/1.0.3) (2024-11-01)

[Full Changelog](https://github.com/OpenVoiceOS/ovos-padatious-pipeline-plugin/compare/1.0.3a1...1.0.3)

**Merged pull requests:**

- Release 1.0.3a1 [\#23](https://github.com/OpenVoiceOS/ovos-padatious-pipeline-plugin/pull/23) ([github-actions[bot]](https://github.com/apps/github-actions))

## [1.0.3a1](https://github.com/OpenVoiceOS/ovos-padatious-pipeline-plugin/tree/1.0.3a1) (2024-10-31)

[Full Changelog](https://github.com/OpenVoiceOS/ovos-padatious-pipeline-plugin/compare/1.0.2...1.0.3a1)

**Merged pull requests:**

- fix:workshop2\_compat [\#22](https://github.com/OpenVoiceOS/ovos-padatious-pipeline-plugin/pull/22) ([JarbasAl](https://github.com/JarbasAl))

## [1.0.2](https://github.com/OpenVoiceOS/ovos-padatious-pipeline-plugin/tree/1.0.2) (2024-10-16)

[Full Changelog](https://github.com/OpenVoiceOS/ovos-padatious-pipeline-plugin/compare/1.0.2a1...1.0.2)

**Merged pull requests:**

- Release 1.0.2a1 [\#20](https://github.com/OpenVoiceOS/ovos-padatious-pipeline-plugin/pull/20) ([github-actions[bot]](https://github.com/apps/github-actions))

## [1.0.2a1](https://github.com/OpenVoiceOS/ovos-padatious-pipeline-plugin/tree/1.0.2a1) (2024-10-16)

[Full Changelog](https://github.com/OpenVoiceOS/ovos-padatious-pipeline-plugin/compare/1.0.1...1.0.2a1)

**Merged pull requests:**

- drop threading [\#19](https://github.com/OpenVoiceOS/ovos-padatious-pipeline-plugin/pull/19) ([JarbasAl](https://github.com/JarbasAl))

## [1.0.1](https://github.com/OpenVoiceOS/ovos-padatious-pipeline-plugin/tree/1.0.1) (2024-10-16)

[Full Changelog](https://github.com/OpenVoiceOS/ovos-padatious-pipeline-plugin/compare/1.0.1a1...1.0.1)

**Merged pull requests:**

- Release 1.0.1a1 [\#18](https://github.com/OpenVoiceOS/ovos-padatious-pipeline-plugin/pull/18) ([github-actions[bot]](https://github.com/apps/github-actions))

## [1.0.1a1](https://github.com/OpenVoiceOS/ovos-padatious-pipeline-plugin/tree/1.0.1a1) (2024-10-16)

[Full Changelog](https://github.com/OpenVoiceOS/ovos-padatious-pipeline-plugin/compare/1.0.0...1.0.1a1)

**Merged pull requests:**

- port tests from core [\#17](https://github.com/OpenVoiceOS/ovos-padatious-pipeline-plugin/pull/17) ([JarbasAl](https://github.com/JarbasAl))

## [1.0.0](https://github.com/OpenVoiceOS/ovos-padatious-pipeline-plugin/tree/1.0.0) (2024-10-16)

[Full Changelog](https://github.com/OpenVoiceOS/ovos-padatious-pipeline-plugin/compare/1.0.0a1...1.0.0)

**Merged pull requests:**

- Release 1.0.0a1 [\#16](https://github.com/OpenVoiceOS/ovos-padatious-pipeline-plugin/pull/16) ([github-actions[bot]](https://github.com/apps/github-actions))

## [1.0.0a1](https://github.com/OpenVoiceOS/ovos-padatious-pipeline-plugin/tree/1.0.0a1) (2024-10-16)

[Full Changelog](https://github.com/OpenVoiceOS/ovos-padatious-pipeline-plugin/compare/0.1.3...1.0.0a1)

**Breaking changes:**

- feat!:pipeline factory [\#15](https://github.com/OpenVoiceOS/ovos-padatious-pipeline-plugin/pull/15) ([JarbasAl](https://github.com/JarbasAl))

## [0.1.3](https://github.com/OpenVoiceOS/ovos-padatious-pipeline-plugin/tree/0.1.3) (2024-10-16)

[Full Changelog](https://github.com/OpenVoiceOS/ovos-padatious-pipeline-plugin/compare/0.1.3a1...0.1.3)

**Merged pull requests:**

- Release 0.1.3a1 [\#14](https://github.com/OpenVoiceOS/ovos-padatious-pipeline-plugin/pull/14) ([github-actions[bot]](https://github.com/apps/github-actions))

## [0.1.3a1](https://github.com/OpenVoiceOS/ovos-padatious-pipeline-plugin/tree/0.1.3a1) (2024-10-16)

[Full Changelog](https://github.com/OpenVoiceOS/ovos-padatious-pipeline-plugin/compare/0.1.2...0.1.3a1)

**Merged pull requests:**

- std lang codes [\#13](https://github.com/OpenVoiceOS/ovos-padatious-pipeline-plugin/pull/13) ([JarbasAl](https://github.com/JarbasAl))

## [0.1.2](https://github.com/OpenVoiceOS/ovos-padatious-pipeline-plugin/tree/0.1.2) (2024-10-15)

[Full Changelog](https://github.com/OpenVoiceOS/ovos-padatious-pipeline-plugin/compare/0.1.2a2...0.1.2)

**Merged pull requests:**

- Release 0.1.2a2 [\#11](https://github.com/OpenVoiceOS/ovos-padatious-pipeline-plugin/pull/11) ([github-actions[bot]](https://github.com/apps/github-actions))

## [0.1.2a2](https://github.com/OpenVoiceOS/ovos-padatious-pipeline-plugin/tree/0.1.2a2) (2024-10-15)

[Full Changelog](https://github.com/OpenVoiceOS/ovos-padatious-pipeline-plugin/compare/0.1.2a1...0.1.2a2)

**Merged pull requests:**

- allow workshop 1.0.0 [\#10](https://github.com/OpenVoiceOS/ovos-padatious-pipeline-plugin/pull/10) ([JarbasAl](https://github.com/JarbasAl))

## [0.1.2a1](https://github.com/OpenVoiceOS/ovos-padatious-pipeline-plugin/tree/0.1.2a1) (2024-10-15)

[Full Changelog](https://github.com/OpenVoiceOS/ovos-padatious-pipeline-plugin/compare/0.1.1a1...0.1.2a1)

**Merged pull requests:**

- fix:license tests [\#6](https://github.com/OpenVoiceOS/ovos-padatious-pipeline-plugin/pull/6) ([JarbasAl](https://github.com/JarbasAl))

## [0.1.1a1](https://github.com/OpenVoiceOS/ovos-padatious-pipeline-plugin/tree/0.1.1a1) (2024-10-14)

[Full Changelog](https://github.com/OpenVoiceOS/ovos-padatious-pipeline-plugin/compare/0.1.0...0.1.1a1)

**Merged pull requests:**

- tests [\#4](https://github.com/OpenVoiceOS/ovos-padatious-pipeline-plugin/pull/4) ([JarbasAl](https://github.com/JarbasAl))

## [0.1.0](https://github.com/OpenVoiceOS/ovos-padatious-pipeline-plugin/tree/0.1.0) (2024-10-14)

[Full Changelog](https://github.com/OpenVoiceOS/ovos-padatious-pipeline-plugin/compare/0.1.0a2...0.1.0)

## [0.1.0a2](https://github.com/OpenVoiceOS/ovos-padatious-pipeline-plugin/tree/0.1.0a2) (2024-10-14)

[Full Changelog](https://github.com/OpenVoiceOS/ovos-padatious-pipeline-plugin/compare/9927485b5fdc8f1fa5b34a01c9011189579f3c9b...0.1.0a2)

**Merged pull requests:**

- feat:semver [\#3](https://github.com/OpenVoiceOS/ovos-padatious-pipeline-plugin/pull/3) ([JarbasAl](https://github.com/JarbasAl))
- feat/opm\_pipeline [\#2](https://github.com/OpenVoiceOS/ovos-padatious-pipeline-plugin/pull/2) ([JarbasAl](https://github.com/JarbasAl))



\* *This Changelog was automatically generated by [github_changelog_generator](https://github.com/github-changelog-generator/github-changelog-generator)*
