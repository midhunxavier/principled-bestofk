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
- [ ] document findings for paper
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
