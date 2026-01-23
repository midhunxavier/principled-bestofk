Analysis: Why the Principled Max@K Approach "Failed"
The Experimental Results (TSP50)
| Algorithm | Greedy Cost | Best-of-128 Cost |
|-----------|------------|------------------|
| POMO (baseline) | 6.45 ± 0.05 | 5.98 ± 0.01 |
| Leader Reward | 6.53 ± 0.05 | 6.01 ± 0.01 |
| MaxK POMO (ours) | 6.73 ± 0.17 | 6.28 ± 0.05 |
The training curve shows maxk_pomo starting higher and not converging - it's stuck around 7.7-8.0 while POMO and Leader Reward converge nicely to ~6.4.
---
Root Cause Analysis
1. The Variance Paradox: Theory vs. Practice
The theoretical derivations (Task 1.1-1.4) prove that SubLOO is an unbiased estimator that eliminates hitchhiking. However, there's a critical gap:
Theoretical assumption violated: The proofs in task1.2/unbiasedness_proof.md assume i.i.d. samples, but POMO uses deterministic multi-start (different starting nodes). From the PRD (line 146-148):
> Task 1 proofs assume i.i.d. trajectories; deterministic multi-start is exchangeable but not strictly i.i.d.
With only n=16 starts and k=8, the samples are not truly i.i.d. - they're deterministic permutations from fixed start nodes. This breaks the unbiasedness guarantee.
2. The Weight Sparsity Problem
SubLOO weights are gap-based: only the max in each subset gets gradient signal, and that weight is R_max - R_second_max. In TSP:
- Tour costs are highly concentrated (std ~0.4 on costs 6.5 = 6% variance)
- Gaps between best and second-best are tiny
- This means SubLOO weights are near-zero most of the time
From task1.3/loo_variance_reduction.md (Section 2.2.1):
> Only the subset's winner receives non-zero gradient weight, and that weight equals the max–second-max gap.
Problem: When rewards are tightly clustered (as in NCO), the gap signal is extremely weak, causing:
- Very small effective learning rate
- High relative variance in the gradient direction
- Slow/no convergence
3. The Sample Size Mismatch
Configuration: n=16 samples, k=8, SubLOO variance reduction.
From the theory:
- SubLOO requires k ≥ 2 ✓
- But effective variance reduction scales as O(1/n) 
- With n=16 and k=8, you have C(16,8) = 12870 subsets, but each sample only contributes to C(15,7) = 6435 of them
The problem: the estimator is unbiased but has HIGH variance when n is small relative to k. The PRD target was n ≥ 2k (line 426: "Use n ≥ 2K if possible"), which is barely satisfied.
4. Hitchhiking Elimination Backfires
This is perhaps the most subtle issue. From task1.4/variance_comparison_analysis.md:
> SubLOO sets Support-term contributions exactly to zero, removing a significant variance source.
But the Support term isn't just noise - it provides dense gradient signal to lower-ranked samples. By eliminating it entirely:
- Only top-k samples get any gradient
- Exploration of the policy space is reduced
- The policy can get stuck in a local optimum
In contrast, Leader Reward's "hitchhiking" actually provides beneficial gradient diffusion that helps the policy explore.
5. Training Instability
Looking at the training curves:
- maxk_pomo has high variance across epochs (oscillating between 7.5-8.1)
- This suggests the gradient estimates are too noisy
- The model never stabilizes into a good region
From the config, no gradient clipping was applied (gradient_clip_val: null). The PRD recommended gradient clipping (Section 9.3), but it wasn't used.
---
Is This "The Truth" or Did We Go Wrong?
Both. Here's my assessment:
What's True (The Hard Reality)
1. Unbiasedness ≠ Good Performance: An unbiased estimator with high variance can be worse than a biased estimator with low variance. Leader Reward's bias is $O(1)$ but its variance is also lower in practice.
2. NCO is Different from LLM Settings: The PKPO/RSPO papers work with binary pass/fail or sparse rewards. NCO has dense, continuous rewards with small gaps. The methods don't transfer directly.
3. SubLOO's Gap-Based Weights Kill the Signal: When rewards are tightly clustered, the max-second_max gap is tiny, making learning extremely slow.
What Went Wrong (Fixable Issues)
1. Too Aggressive Variance Reduction: SubLOO is the most aggressive option. Sample-LOO might work better.
2. n/k Ratio Too Low: With n=16, k=8, we're at the edge. Need n=32 or n=64 for stable gradients.
3. No Entropy Regularization: Without entropy, the policy collapses quickly when only top-k samples get gradients.
4. POMO's Deterministic Starts: The i.i.d. assumption is violated. Need to sample start nodes randomly.
5. No Gradient Clipping: Unstable gradients weren't controlled.
6. Only 10 Epochs: This is far too short. POMO typically trains for 100+ epochs.
---
Recommendations for Next Steps
1. Try Sample-LOO instead of SubLOO: Less aggressive, maintains some "support" gradient signal.
2. Increase n/k ratio: Use n=32, k=4 or n=64, k=8.
3. Add Entropy Regularization: Critical for maintaining exploration when gradient signal is sparse.
4. Use Random Start Nodes: Sample with replacement to satisfy i.i.d. assumption.
5. Normalize/Scale the Weights: The SubLOO weights may need normalization (they sum to ~K * avg_gap, not 1).
6. Longer Training: Run for 100 epochs with proper learning rate scheduling.
7. Hybrid Approach: Combine SubLOO with a baseline POMO term: w_i = λ * SubLOO_weight + (1-λ) * POMO_weight.