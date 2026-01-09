# Examples: Planning with Files in Action

## Example 1: Research Task - Deriving Max@K Gradient Estimator

**User Request:** "Derive an unbiased gradient estimator for Max@K and compare to Leader Reward"

### Loop 1: Create Task Folder and Plan
```bash
mkdir -p docs/tasks/maxk-gradient-estimator
Write docs/tasks/maxk-gradient-estimator/task_plan.md
Write docs/tasks/maxk-gradient-estimator/findings.md
Write docs/tasks/maxk-gradient-estimator/progress.md
```

### docs/tasks/maxk-gradient-estimator/task_plan.md
```markdown
# Task Plan: Unbiased Max@K Gradient Estimator

## Goal
Derive a principled, unbiased gradient estimator for Max@K objective that improves on Leader Reward.

## Current Phase
Phase 1

## Research Context
**Reference:** knowledgebase/llm_context_maxk_rl4co.txt
Key insight: Leader Reward is heuristic; PKPO/RSPO provide principled alternatives.

## Phases
- [ ] Phase 1: Review PKPO/RSPO math ✓
- [ ] Phase 2: Derive CO-specific estimator
- [ ] Phase 3: Implement in RL4CO
- [ ] Phase 4: Benchmark vs Leader Reward
- [ ] Phase 5: Document for paper

## Status
**Currently in Phase 1** - Reviewing existing estimators
```

### Loop 2: Literature Review
```bash
Read docs/tasks/maxk-gradient-estimator/task_plan.md     # Refresh goals
Read knowledgebase/llm_context_maxk_rl4co.txt            # Get context
Write docs/tasks/maxk-gradient-estimator/findings.md     # Store key equations
Edit docs/tasks/maxk-gradient-estimator/task_plan.md     # Mark Phase 1 complete
```

### docs/tasks/maxk-gradient-estimator/findings.md
```markdown
# Findings: Max@K Gradient Estimator

## Key Equations from PKPO/RSPO

### Unbiased maxg@k estimator (PKPO):
ρ^(g)(n,k) = (1 / C(n,k)) Σ_{i=k..n} C(i−1, k−1) * g_i

### Gradient weights:
∇̂^(g) = Σ_i s_i ∇θ log p(x_i | θ)
where s_i is weighted by marginal contribution.

## Gap vs Leader Reward
- Leader Reward: only boosts the single best sample
- PKPO/RSPO: weight ALL samples by their marginal contribution to max@k
- This provides credit assignment across the entire reward distribution

## Key Insight
Can combine POMO's multi-start exploration with RSPO's marginal contribution weights!
```

### Loop 3: Mathematical Derivation
```bash
Read docs/tasks/maxk-gradient-estimator/task_plan.md    # Refresh goals
Read docs/tasks/maxk-gradient-estimator/findings.md     # Get previous work
# ... derivation work ...
Edit docs/tasks/maxk-gradient-estimator/findings.md     # Add new derivations
Edit docs/tasks/maxk-gradient-estimator/task_plan.md    # Mark Phase 2 complete
```

---

## Example 2: Implementation Task - RL4CO Estimator

**User Request:** "Fix the login bug in the authentication module"

### task_plan.md
```markdown
# Task Plan: Fix Login Bug

## Goal
Identify and fix the bug preventing successful login.

## Phases
- [x] Phase 1: Understand the bug report ✓
- [x] Phase 2: Locate relevant code ✓
- [ ] Phase 3: Identify root cause (CURRENT)
- [ ] Phase 4: Implement fix
- [ ] Phase 5: Test and verify

## Key Questions
1. What error message appears?
2. Which file handles authentication?
3. What changed recently?

## Decisions Made
- Auth handler is in src/auth/login.ts
- Error occurs in validateToken() function

## Errors Encountered
- [Initial] TypeError: Cannot read property 'token' of undefined
  → Root cause: user object not awaited properly

## Status
**Currently in Phase 3** - Found root cause, preparing fix
```

---

## Example 3: Feature Development

**User Request:** "Add a dark mode toggle to the settings page"

### The 3-File Pattern in Action

**task_plan.md:**
```markdown
# Task Plan: Dark Mode Toggle

## Goal
Add functional dark mode toggle to settings.

## Phases
- [x] Phase 1: Research existing theme system ✓
- [x] Phase 2: Design implementation approach ✓
- [ ] Phase 3: Implement toggle component (CURRENT)
- [ ] Phase 4: Add theme switching logic
- [ ] Phase 5: Test and polish

## Decisions Made
- Using CSS custom properties for theme
- Storing preference in localStorage
- Toggle component in SettingsPage.tsx

## Status
**Currently in Phase 3** - Building toggle component
```

**notes.md:**
```markdown
# Notes: Dark Mode Implementation

## Existing Theme System
- Located in: src/styles/theme.ts
- Uses: CSS custom properties
- Current themes: light only

## Files to Modify
1. src/styles/theme.ts - Add dark theme colors
2. src/components/SettingsPage.tsx - Add toggle
3. src/hooks/useTheme.ts - Create new hook
4. src/App.tsx - Wrap with ThemeProvider

## Color Decisions
- Dark background: #1a1a2e
- Dark surface: #16213e
- Dark text: #eaeaea
```

**dark_mode_implementation.md:** (deliverable)
```markdown
# Dark Mode Implementation

## Changes Made

### 1. Added dark theme colors
File: src/styles/theme.ts
...

### 2. Created useTheme hook
File: src/hooks/useTheme.ts
...
```

---

## Example 4: Error Recovery Pattern

When something fails, DON'T hide it:

### Before (Wrong)
```
Action: Read config.json
Error: File not found
Action: Read config.json  # Silent retry
Action: Read config.json  # Another retry
```

### After (Correct)
```
Action: Read config.json
Error: File not found

# Update task_plan.md:
## Errors Encountered
- config.json not found → Will create default config

Action: Write config.json (default config)
Action: Read config.json
Success!
```

---

## The Read-Before-Decide Pattern

**Always read your plan before major decisions:**

```
[Many tool calls have happened...]
[Context is getting long...]
[Original goal might be forgotten...]

→ Read task_plan.md          # This brings goals back into attention!
→ Now make the decision       # Goals are fresh in context
```

This is why Manus can handle ~50 tool calls without losing track. The plan file acts as a "goal refresh" mechanism.
