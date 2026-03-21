# Gemini CLI Workflow Mandates

This file contains foundational instructions that take absolute precedence over general workflows.

## 1. Documentation First (Reasoning-to-File)
- **Mandate:** Before executing any `replace`, `write_file`, or `run_shell_command` that modifies the codebase or starts a training process, you **MUST** document your reasoning, logic, and specific plan in `SESSION_NOTES.md`.
- **Purpose:** To provide a clear audit trail of technical decisions and ensure the user can review the logic before implementation.
- **Content:** Include why a specific approach was chosen (e.g., why Beam Search over Greedy), expected outcomes, and any constraints/risks identified.

## 2. Session Continuity
- At the start of every session, read `PROJECT_SUMMARY.md` and `SESSION_NOTES.md` to synchronize with the current state.
- Update `PROJECT_SUMMARY.md` at the end of every significant task.

## 3. Engineering Standards
- Adhere to the established preparation and validation cycles (Research -> Strategy -> Execution).
- For AI tasks, follow the 4-Phase Model Development Workflow (Sanity Check -> Prototype -> Optimization -> Production).
