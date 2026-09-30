
# Agent-assisted engineering workflow

This document defines the workflow for substantial engineering work. The
primary user-facing agent coordinates the process.

## When to use this workflow

Use this workflow when the task is not safely handled as a single,
well-understood, and clearly delimited edit.

Typical triggers include:
- ambiguous or evolving requirements
- multi-step implementation
- changes across multiple components
- meaningful design or architectural decisions
- external research that can affect the approach
- multiple agents or parallel work
- meaningful regression, compatibility, or operational risk

## Durable task state

For substantial work that benefits from resumability, create:

`.agents/work/<TASK-ID-or-short-slug>/`

Use the templates under `.agents/templates/`.

Keep task artifacts concise and current. Do not copy raw agent transcripts into
them, and do not rely on chat history as the only record of material
requirements or decisions.

Typical artifacts:
- `spec.md` for requirements, constraints, acceptance criteria, and non-goals
- `plan.md` when an implementation plan is required
- `decisions.md` for material task-specific decisions when useful
- `reviews/` for review reports worth preserving

Repository-wide architectural or workflow decisions that should outlive the
task belong in `docs/decision_log.md`.

## Writing standard
Write durable task artifacts in clear, direct technical English. 

When creating or updating `spec.md` or `plan.md`, apply the `asd-ste100` skill. Preserve all technical precision, constraints, exceptions, and acceptance criteria.

Do not introduce new terminology when plain existing terms are sufficient. Define necessary project-specific terms when first used.

## 1. Requirements and specification

For work with non-trivial or evolving requirements, iterate with the user until
the desired outcome is sufficiently concrete to proceed.

Create or update `spec.md` when durable requirements are useful. The specification should capture:
- problem / desired outcome
- required behavior
- acceptance criteria
- relevant constraints and compatibility requirements
- non-goals
- unresolved product or scientific decisions

Ask the user about consequential requirements, scope, compatibility, scientific,
or product decisions. Resolve routine implementation questions from repository
evidence and established conventions where possible.

Exit when the task is concrete enough to investigate, plan, implement, or
review without silently inventing requirements or risk missing important criteria.

Ask the user to approve the final `spec.md` before proceeding.


## 2. Discovery

Delegate repository questions to `explorer` and external questions to
`researcher` sub-agents.

Prefer narrow questions over broad "research everything" assignments.

Good exploration questions identify things such as:
- real entrypoints and control/data flow
- existing implementations and extension points
- affected tests, configs, inputs, and outputs
- architectural or compatibility constraints

Good research questions identify things such as:
- current library/framework behavior
- authoritative external documentation
- best practices and standards
- relevant prior art and examples
- existing libraries and tools that can get the job done

The primary agent synthesizes the findings of the sub-agents. Preserve only evidence or conclusions that remain useful to the task.


## 3. Decide the planning level

For any non-trivial implementation task, outline a brief high-level plan for the user before editing.

Create a persistent `plan.md` when one or more of the following apply:
- the task spans multiple components with non-obvious interactions
- there are meaningful architectural or design choices
- sequencing or dependencies matter
- multiple implementation agents may be used
- external research materially affects the implementation approach
- compatibility, migration, rollout, or rollback must be considered
- acceptance criteria require several coordinated implementation steps

For such tasks, follow `.agents/PLANS.md`.

For small, contained implementation tasks, the in-chat plan is sufficient and a persistent `plan.md` is not required.

If major implementation work later becomes necessary, reassess this decision together with the user.

Ask the user to approve the final `plan.md` before proceeding.


## 4. Plan review

When `plan.md` exists for a substantial implementation, have a fresh
`plan_reviewer` sub-agent review it before implementation.

Provide the reviewer with:
- the specification
- the plan
- relevant exploration/research evidence
- repository context needed to challenge important assumptions

The reviewer must not implement the plan.

The primary agent triages reviewer findings as:
- ACCEPT: the finding is valid and should be addressed before proceeding.
- REJECT_WITH_EVIDENCE: the finding is not valid for the current change; record the reason and concrete evidence supporting rejection.
- CLARIFY: the finding may be valid but is too ambiguous or underspecified to act on; return it to the reviewer for a more precise claim.
- ESCALATE: resolving the finding requires a consequential product, scientific, scope, compatibility, or risk decision that belongs to the user or another human owner.
- DEFER: the finding is valid but intentionally excluded from the current scope; record the rationale and create explicit follow-up work if it should not be lost.

DEFER must not be used for a blocking requirement or correctness issue unless the user explicitly accepts the resulting risk or scope change.

Resolve blocking findings before implementation. Escalate consequential requirements, scope, or risk decisions to the user.


## 5. Implementation

Delegate bounded implementation work to `implementer` sub-agents. Use an appropriate number of agents for the breadth and complexity of the task.

Each work assignment should provide:
- objective
- relevant spec/plan/task paths
- constraints and scope
- completion criteria
- required checks

Prefer isolated branches/worktrees for concurrent writers. Do not assign
overlapping write ownership unless explicitly coordinated.

Implementers must:
- inspect relevant repository instructions before editing
- follow the root-level `AGENTS.md` for all work
- follow `docs/coding_guidelines.md`
- follow `docs/test_guidelines.md` and `src/tests/AGENTS.md` when modifying tests
- follow applicable notebook-specific `AGENTS.md` when modifying notebooks
- add or update tests for changed behavior where appropriate
- run the smallest relevant checks plus required pre-commit checks
- report commands run, results, changed files, and unresolved risks

Automated hooks and CI are authoritative for the checks they enforce.


## 6. Code review

Use a fresh `code_reviewer` for substantial code changes, even when the task was small enough not to require persistent `spec.md` or `plan.md` artifacts.

When persistent artifacts exist, provide the reviewer with:
- the specification
- the approved plan, if any
- the complete change
- relevant test/check evidence

When they do not exist, provide:
- the original task/request
- the touched files or complete diff
- any material assumptions or acceptance criteria established in chat
- relevant test/check evidence

The reviewer should inspect enough surrounding code to assess the change in context rather than limiting review strictly to edited lines. The reviewer should consider:
- correctness and regressions
- config, I/O, and validation contracts
- adequacy of tests
- maintainability and unnecessary complexity
- documentation accuracy
- repository conventions in `docs/coding_guidelines.md`
- relevant security concerns

Route findings through the primary agent. Raise issues that require reconsidering or material decisions to the user.

For immediate narrow corrections, resume the original implementer when
practical. For delayed, broad, or assumption-challenging findings, use a fresh
implementer/fixer with the current branch, requirements, and review findings.

After material fixes, rerun affected checks and repeat review as needed.


## 7. PR handoff

Prepare a PR only when the change is coherent and applicable local review/check
gates have been completed, unless remote CI or early collaboration makes an
earlier draft PR useful.

The PR should summarize:
- linked issue/work item when applicable
- relevant spec/plan paths
- behavior changed
- validation performed
- known limitations or follow-up work

PR review and CI are additional integration gates, not substitutes for local
implementation checks and independent review.

Do not create a PR without user approval unless that authority has been
explicitly delegated.

## Decision log

Always summarize and log important changes in `docs/decision_log.md`.
