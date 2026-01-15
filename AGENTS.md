# Repository Guidelines

## Project Structure & Module Organization
This repository is research- and documentation-focused. Key locations:
- `docs/`: core project narrative and tasks (e.g., `docs/PRD.md`, `docs/tasks/task1.1/mathematical_derivation.md`).
- `docs/description.txt`: short project summary and pointers.
- `knowledgebase/`: supporting references, including PDFs in `knowledgebase/papers/` and context notes like `knowledgebase/llm_context_maxk_rl4co.txt`.
- `LICENSE`: project license.

## Build, Test, and Development Commands
There is no build system or runnable application in this repo yet.
- Build: not applicable (documentation-only).
- Test: not applicable (no test framework present).
- Run: not applicable (no runtime entrypoint).

If you introduce scripts later, document the exact commands here.

## Coding Style & Naming Conventions
- Use Markdown for new documents and keep headings structured with `#`, `##`, `###`.
- Follow the existing mathematical style: use LaTeX blocks for equations and tables for comparisons.
- Prefer lowercase, underscore-separated filenames (e.g., `mathematical_derivation.md`).
- For task work, mirror the existing layout in `docs/tasks/taskX.Y/` and add a concise README or Markdown deliverable inside.

## Testing Guidelines
No automated tests exist at the moment. If you add code, include a minimal test plan and note how to run it in the Build/Test section.

## Commit & Pull Request Guidelines
- Commit messages are short and task-oriented (examples in history: `Task1.1`, `v1`, `v2`).
- Keep commits scoped to a single deliverable when possible.
- For pull requests, include a brief summary of what changed, the files touched, and any open questions or follow-up work.

## Knowledgebase & References
- Add new papers to `knowledgebase/papers/` with clear filenames.
- Add new context notes as `.txt` or `.md` in `knowledgebase/`.
- Ensure you have the right to store any PDFs or external materials.
