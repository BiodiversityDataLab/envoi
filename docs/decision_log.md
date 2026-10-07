# Decision log

This log records lasting architectural, workflow, and scientific decisions for envoi, as `AGENTS.md` requires.
Each entry gives the date, the decision, the reason, and the alternatives that were rejected.
Add new entries at the end. Do not edit old entries. When a decision changes, add a new entry that names the old one.

## 2026-10-07 — Agent setup for Codex and Claude Code

These decisions came from the review of the shared agent configuration (`AGENTS.md`, `.agents/`, `.codex/`, `.claude/`).

### One copy of the role instructions

- **Decision:** the `developer_instructions` in `.codex/agents/<role>.toml` are the only copy of each specialist role's instructions.
  The Claude Code wrappers in `.claude/agents/<role>.md` tell the agent to read the matching `.toml` file, and to stop if it cannot read it.
- **Reason:** two copies would drift. Codex gets its instructions injected directly, so its behavior does not change.
- **Rejected:** shared `.agents/roles/<role>.md` files (Codex would then also depend on a file read), and full copies in both tools.

### Five Claude Code roles with fixed tools

- **Decision:** Claude Code has all five roles: `explorer`, `researcher`, `plan_reviewer`, `implementer`, and `code_reviewer`.
  `explorer`, `plan_reviewer`, and `code_reviewer` get only Read, Grep, and Glob. `researcher` also gets WebSearch and WebFetch. `implementer` gets all tools.
- **Reason:** the built-in Claude Code agents do not follow the repository's role rules. The tool lists make the review roles read-only.
- **Consequences:** the read-only roles cannot run commands, so the primary agent gives `code_reviewer` the diff and the check results.
  The Claude Code `researcher` cannot write files, so the primary agent writes research files from its answer.
- **Rejected:** use the built-in `Explore` and `general-purpose` agents for `explorer` and `researcher`. Give `code_reviewer` Bash (it could then change files).

### Models

- **Decision:** the Codex role files keep their pinned model names and reasoning efforts. The Claude Code wrappers use the aliases `opus` (`plan_reviewer`, `code_reviewer`, `implementer`) and `sonnet` (`explorer`, `researcher`), which mirror the Codex tiers.
  The team updates both by hand when a model is replaced.
- **Reason:** the team wants explicit control of the models. Aliases name no version, so they need fewer updates than pinned IDs.
- **Rejected:** `model: inherit` for the Claude Code roles. Review quality would then depend on the model of the session that starts the review.

### Required return headings

- **Decision:** each role's "Return" list starts with "Use these items as section headings, in this order. Write 'none' for an empty item."
- **Reason:** in tests, one Claude Code role and one Codex role each left out part of their output contract. With the line, all roles passed.
- **Rejected:** accept answers with the right content but no headings. One answer had left out content, not only headings.

### Location of the shared skills

- **Decision:** shared skills live in `.agents/skills/`, where Codex finds them. `.claude/skills/<skill>` is a symlink to the same folder, for Claude Code. `AGENTS.md` also gives the skill path.
- **Reason:** each tool discovers skills in a different folder. A symlink keeps one copy.
- **Consequence:** on Windows without symlink support, Claude Code finds the skill only through the path in `AGENTS.md`. This limit was accepted without a Windows test.
- **Rejected:** the real folder in `.claude/skills/`, a second copy of the skill, and no automatic discovery (the earlier location, a top-level .skills folder, which neither tool scans).

### Where implementers work

- **Decision:** one implementer at a time works in the normal checkout, on the current branch.
  For parallel implementers, the primary agent commits first, then creates one worktree per implementer from the current branch with `git worktree add .claude/worktrees/<name> <branch>`.
  Implementers do not use the worktree isolation of the Claude Code Agent tool.
- **Reason:** that isolation starts from `origin/main`, so it does not contain the current branch or uncommitted work.
- **Rejected:** push work to `main` before each implementer, and drop parallel implementation.

### Shared Claude Code permissions

- **Decision:** `.claude/settings.json` allows these exact commands without a prompt: `pytest -m "not gee"`, `ruff check src tests`, `black --check src tests`, `pre-commit run --files ...`, `git status`, `git diff`, `git diff --stat`, `git diff --cached`, and `git log --oneline`.
  It denies reads of `credentials/`, `.env`, `~/.config/envoi/`, and `~/AppData/Roaming/envoi/`.
- **Reason:** fewer permission prompts for routine checks, and a second barrier for the rule "do not read credentials" in `AGENTS.md`.
- **Limits:**
  - The deny rules worked in sessions started in the repository root. In a session started in `src/`, reads of `../credentials/` asked for permission instead of being denied. The cause was not found.
  - The deny rules do not cover a file named by `ENVOI_EE_CREDENTIALS` in another folder, or every possible way to read a file.
  - `pre-commit run --files ... -c <file>` can run any hook of another configuration file. An agent must first write that file.
  - Codex ignores `sandbox_mode` in role files. A Codex role is read-only only when its session starts read-only (`codex --sandbox read-only`).
- **Rejected:** wildcard rules such as `pytest -m "not gee" *`, because a second `-m gee` option selects the live Earth Engine tests.
  A hook that checks `pytest` options, because it adds a script to maintain.
