---
name: conventional-commit
description: "Write git commit messages that follow the Conventional Commits spec: type(scope): summary, plus body and footers."
---

# Conventional Commit Messages

Use this skill whenever you are about to write or revise a git commit message.
It produces messages that follow the [Conventional Commits](https://www.conventionalcommits.org)
specification so history stays machine-readable and changelogs/releases can be
derived automatically.

## Structure

```
<type>(<scope>)<!>: <subject>

<body>

<footer>
```

- **type** (required): the kind of change. One of:
  - `feat` — a new feature
  - `fix` — a bug fix
  - `docs` — documentation only
  - `style` — formatting, no code-behavior change
  - `refactor` — code change that is neither a feature nor a fix
  - `perf` — a performance improvement
  - `test` — adding or fixing tests
  - `build` — build system or dependencies
  - `ci` — CI configuration
  - `chore` — maintenance that doesn't touch src or tests
  - `revert` — reverts a previous commit
- **scope** (optional): a noun in parentheses naming the affected area, e.g.
  `feat(parser):`. Omit it if the change is broad.
- **`!`** (optional): place a `!` before the colon to flag a breaking change,
  e.g. `feat(api)!: drop deprecated field`.
- **subject** (required): imperative mood, lower-case start, no trailing period,
  ideally ≤ 50 characters. "add", not "added" or "adds".
- **body** (optional): explain *what* and *why*, not *how*. Wrap at ~72 columns.
  Separate from the subject with one blank line.
- **footer** (optional): metadata such as `BREAKING CHANGE: <description>` or
  issue references like `Closes #123`. Separate from the body with a blank line.

## Rules

1. The header line is `type(scope): subject` with a single space after the colon.
2. Use the imperative mood for the subject ("fix", "add", "remove").
3. Keep the subject concise; move detail into the body.
4. Any breaking change MUST be signalled either by `!` in the header or a
   `BREAKING CHANGE:` footer (or both).
5. One logical change per commit — if the subject needs "and", split the commit.

## Examples

```
feat(auth): add refresh-token rotation

Rotate the refresh token on every use and invalidate the previous one so a
leaked token cannot be replayed.

Closes #482
```

```
fix(parser): handle empty frontmatter without crashing
```

```
refactor(registry)!: return ParsedFrontmatter instead of AgentSpec

BREAKING CHANGE: callers must build their own spec from the parsed mapping.
```
