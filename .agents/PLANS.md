
# Implementation plan standard

Use this document only when `.agents/WORKFLOW.md` determines that a written
implementation plan is required.

A plan is a living, self-contained engineering artifact. Another agent should be able to continue the task using the repository plus the plan without relying on the original conversation.

Keep the plan current as material facts and decisions change. Prefer concrete
repository references and observable completion criteria over vague prose.

## Writing style

Apply the `asd-ste100` skill (`.agents/skills/asd-ste100/SKILL.md`) when writing or revising the plan.

Use short, direct sentences and consistent terminology. Prefer concrete
repository names, files, functions, and observable outcomes over abstract
labels or invented process terminology.

Do not simplify away technical constraints or important qualifiers.

## Required sections

### Goal and scope

What the implementation must accomplish and what is explicitly out of scope.

### Requirements reference

Link to the authoritative `spec.md` or equivalent requirements source. Repeat
only what is needed to understand the technical plan.

### Current-state findings

Summarize relevant code paths, constraints, dependencies, and external behavior
discovered during exploration/research.

### Proposed approach

Describe the chosen implementation strategy and why it fits the existing
system.

### Alternatives and decisions

Record only consequential alternatives. State the selected option and why.

### Implementation tasks

Use ordered, bounded tasks. Each task should state:
- objective
- affected areas
- dependencies
- observable completion condition
- required tests/checks
- whether it is safe to run in parallel

### Validation strategy

Map acceptance criteria to tests, commands, outputs, or other observable
evidence.

### Compatibility, migration, rollout, rollback

Include when relevant. Omit or state not applicable when genuinely unnecessary.

### Risks and open questions

Record known uncertainties, implementation risks, and decisions still required.

### Progress

Track meaningful milestones for long-running or multi-step tasks, not every command or subagent action.

### Review status

Record the plan-review verdict and any unresolved findings.
