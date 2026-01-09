---
name: planning-with-files
version: "3.0.0"
description: Implements Manus-style file-based planning for complex research and implementation tasks. Creates task folders in docs/tasks/ with task_plan.md, findings.md, and progress.md. Use when starting complex multi-step tasks, research projects, or any task requiring >5 tool calls. Specialized for Max@K/Best-of-K RL4CO research.
allowed-tools: Read, Write, Edit, Bash, Glob, Grep, WebFetch, WebSearch
hooks:
  PreToolUse:
    - matcher: "Write|Edit|Bash"
      hooks:
        - type: command
          command: "find docs/tasks -name 'task_plan.md' -exec head -30 {} \\; 2>/dev/null || true"
  Stop:
    - hooks:
        - type: command
          command: "${CLAUDE_PLUGIN_ROOT}/scripts/check-complete.sh"
---

# Planning with Files

Work like Manus: Use persistent markdown files as your "working memory on disk."

## Project Context: Max@K / Best-of-K RL4CO Research

This project focuses on developing **unbiased and variance-reduced estimators for Max@K / Best-of-K RL** in combinatorial optimization. Key goals:

- **Derive estimators cleanly** from first principles (PKPO/RSPO-style)
- **Show lower variance and faster convergence** compared to Leader Reward
- **Generalize across multiple RL4CO tasks** (TSP, VRP, OP, etc.)

**Reference:** See `knowledgebase/llm_context_maxk_rl4co.txt` for complete technical context.

## Quick Start

Before ANY complex task:

1. **Create a task folder** — `docs/tasks/[task-name]/`
2. **Create `task_plan.md`** — See [templates/task_plan.md](templates/task_plan.md)
3. **Create `findings.md`** — See [templates/findings.md](templates/findings.md)
4. **Create `progress.md`** — See [templates/progress.md](templates/progress.md)
5. **Re-read plan before decisions** — Refreshes goals in attention window
6. **Update after each phase** — Mark complete, log errors

## The Core Pattern

```
Context Window = RAM (volatile, limited)
Filesystem = Disk (persistent, unlimited)

→ Anything important gets written to disk.
```

## Task Folder Structure

Each task creates a dedicated folder in `docs/tasks/`:

```
docs/tasks/
└── [task-name]/
    ├── task_plan.md     # Phases, progress, decisions
    ├── findings.md      # Research, discoveries
    └── progress.md      # Session log, test results
```

## File Purposes

| File | Purpose | When to Update |
|------|---------|----------------|
| `docs/tasks/[task]/task_plan.md` | Phases, progress, decisions | After each phase |
| `docs/tasks/[task]/findings.md` | Research, discoveries | After ANY discovery |
| `docs/tasks/[task]/progress.md` | Session log, test results | Throughout session |
| `knowledgebase/*.txt` | Persistent research context | When learning new domain info |

## Critical Rules

### 1. Create Plan First
Never start a complex task without `task_plan.md`. Non-negotiable.

### 2. The 2-Action Rule
> "After every 2 view/browser/search operations, IMMEDIATELY save key findings to text files."

This prevents visual/multimodal information from being lost.

### 3. Read Before Decide
Before major decisions, read the plan file. This keeps goals in your attention window.

### 4. Update After Act
After completing any phase:
- Mark phase status: `in_progress` → `complete`
- Log any errors encountered
- Note files created/modified

### 5. Log ALL Errors
Every error goes in the plan file. This builds knowledge and prevents repetition.

```markdown
## Errors Encountered
| Error | Attempt | Resolution |
|-------|---------|------------|
| FileNotFoundError | 1 | Created default config |
| API timeout | 2 | Added retry logic |
```

### 6. Never Repeat Failures
```
if action_failed:
    next_action != same_action
```
Track what you tried. Mutate the approach.

## The 3-Strike Error Protocol

```
ATTEMPT 1: Diagnose & Fix
  → Read error carefully
  → Identify root cause
  → Apply targeted fix

ATTEMPT 2: Alternative Approach
  → Same error? Try different method
  → Different tool? Different library?
  → NEVER repeat exact same failing action

ATTEMPT 3: Broader Rethink
  → Question assumptions
  → Search for solutions
  → Consider updating the plan

AFTER 3 FAILURES: Escalate to User
  → Explain what you tried
  → Share the specific error
  → Ask for guidance
```

## Read vs Write Decision Matrix

| Situation | Action | Reason |
|-----------|--------|--------|
| Just wrote a file | DON'T read | Content still in context |
| Viewed image/PDF | Write findings NOW | Multimodal → text before lost |
| Browser returned data | Write to file | Screenshots don't persist |
| Starting new phase | Read plan/findings | Re-orient if context stale |
| Error occurred | Read relevant file | Need current state to fix |
| Resuming after gap | Read all planning files | Recover state |

## The 5-Question Reboot Test

If you can answer these, your context management is solid:

| Question | Answer Source |
|----------|---------------|
| Where am I? | Current phase in task_plan.md |
| Where am I going? | Remaining phases |
| What's the goal? | Goal statement in plan |
| What have I learned? | findings.md |
| What have I done? | progress.md |

## When to Use This Pattern

**Use for:**
- Multi-step tasks (3+ steps)
- Research tasks (e.g., deriving Max@K estimators)
- Building/creating projects (e.g., RL4CO implementations)
- Tasks spanning many tool calls
- Experiments and benchmarking
- Paper writing and analysis

**Skip for:**
- Simple questions
- Single-file edits
- Quick lookups

## Templates

Copy these templates to start:

- [templates/task_plan.md](templates/task_plan.md) — Phase tracking
- [templates/findings.md](templates/findings.md) — Research storage
- [templates/progress.md](templates/progress.md) — Session logging

## Scripts

Helper scripts for automation:

- `scripts/init-session.sh` — Initialize all planning files
- `scripts/check-complete.sh` — Verify all phases complete

## Advanced Topics

- **Manus Principles:** See [reference.md](reference.md)
- **Real Examples:** See [examples.md](examples.md)

## Anti-Patterns

| Don't | Do Instead |
|-------|------------|
| Use TodoWrite for persistence | Create task_plan.md file |
| State goals once and forget | Re-read plan before decisions |
| Hide errors and retry silently | Log errors to plan file |
| Stuff everything in context | Store large content in files |
| Start executing immediately | Create plan file FIRST |
| Repeat failed actions | Track attempts, mutate approach |
