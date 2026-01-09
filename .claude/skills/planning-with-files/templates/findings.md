# Findings & Decisions

## Requirements
<!-- Captured from user request -->
-

## Literature Review
<!-- Key papers and insights for Max@K/Best-of-K RL4CO -->
| Paper | Key Contribution | Relevance |
|-------|------------------|-----------|
| POMO (2020) | Shared baseline, multi-start | Foundation for multiple samples |
| Leader Reward (2024) | Boost leader trajectory | Heuristic baseline to beat |
| PKPO (2025) | Unbiased maxg@k estimator | Mathematical foundation |
| RSPO (2025) | Marginal contribution weights | Hitchhiking fix |

## Key Equations
<!-- Mathematical formulations from papers -->
### REINFORCE baseline:
∇_θ E_{τ}[R(τ)] = E_{τ}[(R(τ) − b(s)) ∇_θ log π_θ(τ)]

### POMO shared baseline:
b_i(s) = (1/N) Σ_j R(τ^j)

### PKPO maxg@k estimator (unbiased):
ρ^(g)(n,k) = (1 / C(n,k)) Σ_{i=k..n} C(i−1, k−1) * g_i

### PKPO gradient weights:
∇̂^(g) = Σ_i s_i ∇θ log p(x_i | θ)

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

## Gap Analysis
<!-- What prior work DID vs DID NOT do -->
| Approach | Did | Did Not |
|----------|-----|---------|
| POMO (2020) | Multi-start exploration, shared baseline | Train with Max@K objective |
| Leader Reward (2024) | Boost leader during training | Unbiased estimator derivation |
| PKPO/RSPO (2025) | Principled maxg@k/max@k gradients | CO-specific integration |

## Resources
<!-- URLs, file paths, API references -->
- knowledgebase/llm_context_maxk_rl4co.txt (primary reference)
- RL4CO docs: https://rl4co.ai4co.org/

## Visual/Browser Findings
<!-- CRITICAL: Update after every 2 view/browser operations -->
<!-- Multimodal content must be captured as text immediately -->
-

---
*Update this file after every 2 view/browser/search operations*
*This prevents visual information from being lost*
