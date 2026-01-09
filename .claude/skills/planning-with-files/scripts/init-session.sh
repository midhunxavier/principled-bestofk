#!/bin/bash
# Initialize planning files for a new task in docs/tasks/
# Usage: ./init-session.sh <task-name>
# Example: ./init-session.sh maxk-gradient-estimator

set -e

if [ -z "$1" ]; then
    echo "Usage: ./init-session.sh <task-name>"
    echo "Example: ./init-session.sh maxk-gradient-estimator"
    exit 1
fi

TASK_NAME="$1"
TASK_DIR="docs/tasks/$TASK_NAME"
DATE=$(date +%Y-%m-%d)

echo "Initializing task folder: $TASK_DIR"

# Create task directory
mkdir -p "$TASK_DIR"

# Create task_plan.md if it doesn't exist
if [ ! -f "$TASK_DIR/task_plan.md" ]; then
    cat > "$TASK_DIR/task_plan.md" << 'EOF'
# Task Plan: [Brief Description]

## Goal
[One sentence describing the end state]

## Current Phase
Phase 1

## Research Context
**Project:** Unbiased/Variance-Reduced Estimators for Max@K/Best-of-K RL4CO
**Reference:** See `knowledgebase/llm_context_maxk_rl4co.txt`

Key research objectives:
- Derive principled estimators for Max@K objective gradient
- Compare against Leader Reward baseline
- Validate on RL4CO benchmarks (TSP, VRP, OP, etc.)

## Phases

### Phase 1: Requirements & Literature Review
- [ ] Understand task objective
- [ ] Review relevant papers (POMO, Leader Reward, PKPO, RSPO)
- [ ] Document key equations and insights in findings.md
- [ ] Identify gaps and opportunities
- **Status:** in_progress

### Phase 2: Mathematical Derivation
- [ ] Derive estimator from first principles
- [ ] Prove unbiasedness or characterize bias
- [ ] Analyze variance properties
- [ ] Document derivations in findings.md
- **Status:** pending

### Phase 3: Implementation Design
- [ ] Define technical approach
- [ ] Map math to RL4CO code structure
- [ ] Plan integration with existing baselines
- [ ] Document design decisions with rationale
- **Status:** pending

### Phase 4: Implementation
- [ ] Implement estimator in RL4CO framework
- [ ] Write unit tests for gradient computation
- [ ] Integrate with training loop
- [ ] Code review and cleanup
- **Status:** pending

### Phase 5: Experimentation
- [ ] Define experiment protocol
- [ ] Run baselines (POMO, Leader Reward)
- [ ] Run proposed estimator
- [ ] Collect metrics: convergence speed, final performance, variance
- **Status:** pending

### Phase 6: Analysis & Documentation
- [ ] Analyze results vs Leader Reward
- [ ] Create visualizations
- [ ] Document findings for paper
- [ ] Deliver summary to user
- **Status:** pending

## Key Questions
1. [What specific Max@K variant are we optimizing?]
2. [What baselines must we beat?]
3. [Which RL4CO tasks to evaluate on?]

## Decisions Made
| Decision | Rationale |
|----------|-----------|
|          |           |

## Errors Encountered
| Error | Attempt | Resolution |
|-------|---------|------------|
|       | 1       |            |

## Comparisons to Track
| Metric | Leader Reward | Our Estimator | Notes |
|--------|---------------|---------------|-------|
| Bias   |               |               |       |
| Variance|              |               |       |
| Convergence Speed |    |               |       |
| Final Performance |    |               |       |

## Notes
- Update phase status as you progress: pending → in_progress → complete
- Re-read this plan before major decisions (attention manipulation)
- Log ALL errors - they help avoid repetition
- Reference `knowledgebase/llm_context_maxk_rl4co.txt` for technical context
EOF
    echo "Created $TASK_DIR/task_plan.md"
else
    echo "$TASK_DIR/task_plan.md already exists, skipping"
fi

# Create findings.md if it doesn't exist
if [ ! -f "$TASK_DIR/findings.md" ]; then
    cat > "$TASK_DIR/findings.md" << 'EOF'
# Findings & Decisions

## Requirements
<!-- Captured from user request -->
-

## Literature Review
<!-- Key papers and insights -->
| Paper | Key Contribution | Relevance |
|-------|------------------|-----------|
| POMO (2020) | Shared baseline, multi-start | Foundation for multiple samples |
| Leader Reward (2024) | Boost leader trajectory | Heuristic baseline to beat |
| PKPO (2025) | Unbiased maxg@k estimator | Mathematical foundation |
| RSPO (2025) | Marginal contribution weights | Hitchhiking fix |

## Key Equations
<!-- Mathematical formulations -->
-

## Research Findings
<!-- Key discoveries during exploration -->
-

## Technical Decisions
<!-- Decisions made with rationale -->
| Decision | Rationale |
|----------|-----------|
|          |           |

## Issues Encountered
<!-- Errors and how they were resolved -->
| Issue | Resolution |
|-------|------------|
|       |            |

## Resources
<!-- URLs, file paths, API references -->
- knowledgebase/llm_context_maxk_rl4co.txt
-

## Visual/Browser Findings
<!-- CRITICAL: Update after every 2 view/browser operations -->
<!-- Multimodal content must be captured as text immediately -->
-

---
*Update this file after every 2 view/browser/search operations*
*This prevents visual information from being lost*
EOF
    echo "Created $TASK_DIR/findings.md"
else
    echo "$TASK_DIR/findings.md already exists, skipping"
fi

# Create progress.md if it doesn't exist
if [ ! -f "$TASK_DIR/progress.md" ]; then
    cat > "$TASK_DIR/progress.md" << EOF
# Progress Log

## Task: $TASK_NAME
## Session: $DATE

### Phase 1: Requirements & Literature Review
- **Status:** in_progress
- **Started:** $DATE
- Actions taken:
  -
- Files created/modified:
  -

### Phase 2: Mathematical Derivation
- **Status:** pending
- Actions taken:
  -

### Phase 3: Implementation Design
- **Status:** pending
- Actions taken:
  -

### Phase 4: Implementation
- **Status:** pending
- Actions taken:
  -

### Phase 5: Experimentation
- **Status:** pending
- Actions taken:
  -

### Phase 6: Analysis & Documentation
- **Status:** pending
- Actions taken:
  -

## Test Results

| Test | Input | Expected | Actual | Status |
|------|-------|----------|--------|--------|
|      |       |          |        |        |

## Experiment Results

| Experiment | Baseline | Proposed | Improvement |
|------------|----------|----------|-------------|
|            |          |          |             |

## Error Log

<!-- Keep ALL errors - they help avoid repetition -->
| Timestamp | Error | Attempt | Resolution |
|-----------|-------|---------|------------|
|           |       | 1       |            |

## 5-Question Reboot Check
<!-- If you can answer these, context is solid -->
| Question | Answer |
|----------|--------|
| Where am I? | Phase X |
| Where am I going? | Remaining phases |
| What's the goal? | [goal statement] |
| What have I learned? | See findings.md |
| What have I done? | See above |

---
*Update after completing each phase or encountering errors*
EOF
    echo "Created $TASK_DIR/progress.md"
else
    echo "$TASK_DIR/progress.md already exists, skipping"
fi

echo ""
echo "Task folder initialized: $TASK_DIR"
echo "Files created:"
echo "  - $TASK_DIR/task_plan.md"
echo "  - $TASK_DIR/findings.md"
echo "  - $TASK_DIR/progress.md"
echo ""
echo "Reference: knowledgebase/llm_context_maxk_rl4co.txt"
