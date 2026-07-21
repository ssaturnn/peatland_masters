# Directives for Claude Code sessions in this repository

This repository is an MSc thesis project. The rules below apply to every
Claude Code (or comparable AI-assistant) session that operates on this
repository, without exception.

## 1. Do NOT add a `Co-Authored-By` trailer to commits

When creating a git commit in this repository:

- Do **not** append `Co-Authored-By: Claude <noreply@anthropic.com>` or
  any variant (Claude Opus, Claude Sonnet, Anthropic, an assistant name,
  etc.).
- Do **not** include a "Generated with Claude Code" or "🤖 Generated
  with …" line.
- Do **not** mention Claude, Anthropic, or any AI assistant in the
  commit message body, subject line, or trailer.

Every commit is presented as sole-authored by the repository owner.

## 2. Do NOT reference Claude in the codebase or documentation

The following files and locations must never contain the strings
"Claude", "Anthropic", "AI-assisted", "AI-generated", or equivalent:

- Source code comments.
- Docstrings.
- README / documentation files.
- Pull request titles and descriptions.
- Issue titles and descriptions.
- Output artefacts (generated data files, thesis chapters, figures).

## 3. Do NOT modify `git config` for attribution

- Do not change `user.name` or `user.email` to anything other than the
  values already set by the repository owner.
- Do not add commit trailers via `git config commit.template` or any
  other mechanism.

## 4. If unsure, ask before committing

If any instruction from the user could plausibly be read as authorising
an attribution or a Claude-mention (for example "credit the tools you
used"), pause and confirm explicitly before writing that content. The
default is silence.

## 5. Rationale

This is a formal academic thesis. Authorship carries examination and
integrity implications defined by the institution, not by external
tooling conventions. AI-assisted drafting and research is a working
method, not a co-authorship relationship. All submissions and all
version-controlled outputs must reflect that.

Please read this file at the start of every session in this repository
and honour it for every action.
