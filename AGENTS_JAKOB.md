# AGENTS.md

## Scope

- Applies to the entire repository unless a subdirectory-specific `AGENTS.md` provides more specific instructions.
- Prefer repo-local conventions and guidelines over generic agent behavior.
- When working in a subdirectory, read and follow any applicable local `AGENTS.md` before making changes.

## Goals of this project

The goal of this project is to provide a robust, flexible and reproducible machine learning workflow for researchers working on spatial biodiversity models.
- Robust: The code should be accurate, reliable, validated and well-tested so that users can trust the output.
- Flexible: It should be easy to extend and adapt the codebase, e.g. implementing new biodiversity metrics, features, evaluation metrics or models.
- Reproducible: The settings and exact code used for every run should be documented through appropriate logs, metadata and file exports.

## Environment and commands

- Create the environment with `conda env create -f environment.yaml` and activate `sbm_pipe`.
- Add contributor tools with `conda env update -n sbm_pipe -f environment-dev.yaml`.
- Install commit hooks with `pre-commit install`.
- Run hooks for changed files with `pre-commit run --files <path1> <path2>`.
- Run hooks across the repo with `pre-commit run --all-files`.
- Run pipeline entrypoints with `python -m src.dags.dags <dag_name>`.

## Repository map

- Main pipeline code lives under `src/`.
- Pipeline entrypoints live under `src/dags/`.
- Data ingestion and preprocessing code lives under `src/data/`.
- Biodiversity metrics and feature code lives under `src/features/`.
- Model training and validation code lives under `src/models/`.
- Shared path definitions live in `src/paths.py`.
- Shared utilities live under `src/utils/`.
- Runtime validation helpers live under `src/validation/`.
- Automated tests and fixtures live under `src/tests/`.
- User-facing documentation lives in `README.md`, `CONTRIBUTING.md`, and `docs/`.

Important repository guidance:
- Follow `docs/coding_guidelines.md` for code and documentation conventions.
- Follow `docs/test_guidelines.md` for testing conventions.
- When modifying tests, also follow `src/tests/AGENTS.md`.
- When modifying notebooks, follow the applicable notebook-specific `AGENTS.md`.

# Communication and writing

- Do not reference things that the user or another agent has not seen without explanation / definition. Name the thing, not the label.
- Prefer plain, direct language. Avoid invented jargon and unnecessary abstractions.
- Communicate in your own words, not that of sub-agents you communicated with.

Use the `asd-ste100` skill for durable technical artifacts and inter-agent
handoffs when clarity and unambiguous wording matter. In particular, apply it when writing or revising:
- task specifications (`spec.md`)
- implementation plans (`plan.md`)
- agent handoffs
- review findings

## Core working principles

Based on [andrej-karpathy-skills](https://github.com/forrestchang/andrej-karpathy-skills/tree/main).

### 1. Think before coding
**Don't assume. Don't hide confusion. Surface tradeoffs.**

- State material assumptions explicitly. If options could materially affect behavior, surface them rather than choosing silently.
- Prefer simpler approaches when they satisfy the requirements.
- Ask the user when there is ambiguity about requirements, scope, compatibility, or other consequential decisions.
- Resolve routine implementation questions from the repository, existing conventions, and evidence where possible.

### 2. Simplicity first
**Minimum code that solves the problem. Nothing speculative.**

- No features beyond what was requested.
- No abstractions for single-use code without a concrete need.
- No speculative flexibility or configurability.
- Reuse existing components and patterns before introducing new ones.
- If you write 200 lines and it could be 50, rewrite it.

Ask yourself: "Would a senior engineer say this is overcomplicated?" If yes, simplify.

### 3. Surgical changes
**Touch only what you must. Clean up only your own mess.**

- Do not refactor or reformat unrelated code.
- Match existing repository conventions.
- Mention unrelated problems rather than fixing them opportunistically.
- Remove code made obsolete by your own changes.
- Every changed line should have a clear relationship to the requested work.

### 4. Goal-driven execution
**Define success criteria. Loop until verified.**

- Translate requests into testable outcomes before substantial implementation.
- Prefer tests, commands, generated outputs, or other observable evidence over subjective claims that something "works."

## General implementation rules

For any code change:
- Inspect the relevant entrypoints, configs, tests, and input/output paths before editing.
- Prefer the simplest change that satisfies the requirement and reuse existing components before introducing new ones.
- Avoid abstractions and helper methods unless they serve a clear purpose.
- Follow `docs/coding_guidelines.md` as the source of truth for code and documentation patterns.
- Update related documentation and environment/config files only when the change affects them.
- Record important decisions and changes in `docs/decision_log.md`. These include architectural and workflow changes, and important logical or science-related decisions.


## Validation and testing

- Start with the smallest check that directly exercises the changed behavior.
- Add or update tests for changed core logic where appropriate.
- Run `pre-commit run --files <changed paths>` on modified files.
- Follow `docs/test_guidelines.md` and `src/tests/AGENTS.md` when modifying tests.
- If required end-to-end or task-level validation cannot be run in the current environment, state that explicitly and ask the user to run it.


## Code review

When reviewing code:
- Check the implementation against the stated requirements and relevant documentation.
- Inspect the relevant entrypoints, configs, I/O contracts, validation, and tests before drawing conclusions.
- First prioritize functional bugs, requirement violations, and regression risks, and only then stylistic concerns.
- Identify unnecessary complexity when it can be simplified without changing behavior.
- Follow `docs/coding_guidelines.md` for code and documentation conventions.
- Report findings in severity order with concrete locations, evidence, and suggestions.

## Engineering workflow and agent coordination

- Handle small, well-defined changes directly in the current context.
- For non-trivial changes, briefly state the intended approach and material assumptions before editing.
- For substantial work, follow `.agents/WORKFLOW.md`. That workflow determines whether a durable implementation plan is required.

The primary user-facing agent owns coordination and keeps its context focused on requirements, decisions, planning, synthesis, and user interaction.

Delegate substantial specialist or context-heavy work to:
- `explorer` — repository investigation
- `researcher` — external research
- `plan_reviewer` — independent plan review
- `implementer` — code changes and associated tests
- `code_reviewer` — independent implementation review

Subagent findings are inputs to the primary agent; the primary agent remains responsible for synthesis, decisions, and workflow state.

Follow `.agents/WORKFLOW.md` for planning, delegation, handoff, artifact, and review
procedures.

Leverage parallel delegation for independent read-heavy work and implementation that can be clearly separated. Avoid parallel writing to overlapping code unless explicitly planned.

Each concurrent implementation agent must work in its own branch/worktree. Prefer isolated worktrees for substantial implementation tasks generally.


## Safety constraints

- Do not move large datasets into the repository.
- Avoid destructive Git commands unless explicitly requested.
- Do not weaken tests, validation, linting, or safety checks merely to make a change pass.
- Suggest commits and pull requests when appropriate, but wait for user approval before creating them unless the user has explicitly delegated that authority.
