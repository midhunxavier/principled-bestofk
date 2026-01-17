# Principled Best-of-K RL Agent Guidelines

This document provides context and strict guidelines for AI agents (and human developers) working in the `principled-bestofk` repository. This project focuses on developing unbiased, variance-reduced gradient estimators for Max@K objectives in Neural Combinatorial Optimization (NCO), building on the RL4CO library.

## 1. Project Context & Architecture

- **Goal**: Derive and implement "Principled Max@K" policy gradients to replace heuristic "Leader Reward" baselines.
- **Key Concepts**: Max@K Objective, Unbiased Estimators, Leave-One-Out (LOO) Variance Reduction, Combinatorial Optimization (TSP, CVRP).
- **Stack**: Python 3.10+, PyTorch 2.0+, RL4CO (based on TorchRL/Lightning).
- **Status**: Research phase. Code infrastructure is being built out following `docs/PRD.md`.

### Intended Repository Structure
Agents should adhere to this structure when creating new files:
```
principled-bestofk/
├── code/                 # Packaged code and dependencies
│   ├── requirements.txt  # Dependency list (mirrors root)
│   └── src/              # Source code
│       ├── estimators/   # Core math implementations
│       ├── algorithms/   # RL algorithms (MaxK REINFORCE)
│       └── experiments/  # Training scripts
├── docs/                 # Research narrative, proofs, PRD
├── src/                  # Source code (to be created)
│   ├── estimators/       # Core math implementations
│   ├── algorithms/       # RL algorithms (MaxK REINFORCE)
│   └── experiments/      # Training scripts
├── tests/                # Unit tests (pytest)
├── notebooks/            # Analysis and derivation validation
└── AGENTS.md             # This file
```

## 2. Build, Test, and Lint Commands

Although this repository started as documentation-only, it is transitioning to code. Use the following standard commands. If configuration files (`pyproject.toml`, `requirements.txt`) are missing, assume standard Python conventions or ask the user.

### Environment Management
- **Dependency Manager**: Prefer `uv` or `pip`.
- **Install**: `pip install -r code/requirements.txt` (preferred) or `pip install -r requirements.txt` or `pip install torch rl4co`.

### Testing
- **Framework**: `pytest`
- **Run All Tests**:
  ```bash
  pytest
  ```
- **Run Single Test File**:
  ```bash
  pytest tests/test_estimators.py
  ```
- **Run Single Test Method**:
  ```bash
  pytest tests/test_estimators.py::test_unbiasedness -v
  ```
- **Constraint**: *Always* create a test file when creating a new module.

### Linting & Formatting
- **Linter**: `ruff` (preferred) or `flake8`.
- **Formatter**: `ruff format` or `black`.
- **Type Checking**: `mypy`.

**Standard Command Sequence for Agents**:
```bash
# Before committing code changes:
ruff check . --fix  # Fix lint errors
ruff format .       # Enforce style
mypy src/           # Check types
pytest              # Verify correctness
```

## 3. Code Style Guidelines

### Python Code
- **Style**: Follow PEP 8.
- **Formatter**: Use Black-compatible formatting (line length 88 or 100).
- **Type Hints**: **Mandatory** for function signatures.
  ```python
  # Good
  def compute_weights(rewards: torch.Tensor, k: int) -> torch.Tensor:
      ...
  
  # Bad
  def compute_weights(rewards, k):
      ...
  ```
- **Docstrings**: Use **Google-style** docstrings. Include `Args`, `Returns`, and `Raises`.
  ```python
  def loo_baseline(values: torch.Tensor) -> torch.Tensor:
      """Computes Leave-One-Out baseline for variance reduction.

      Args:
          values: Input tensor of shape (batch_size, n_samples).

      Returns:
          Tensor of same shape with baseline subtracted.
      """
      ...
  ```
- **Imports**:
  - Group imports: Standard library, Third-party (torch, numpy), Local `src` imports.
  - Use absolute imports for local modules: `from src.estimators.maxk import ...`

### Naming Conventions
- **Variables/Functions**: `snake_case` (e.g., `compute_loss`, `n_samples`).
- **Classes**: `PascalCase` (e.g., `MaxKEstimator`, `PolicyGradient`).
- **Constants**: `UPPER_CASE` (e.g., `DEFAULT_K`, `EPSILON`).
- **Tensors**: Include shape hints in comments if complex:
  ```python
  log_probs = ... # [batch_size, n_samples]
  ```

### Mathematical Implementation
- **Vectorization**: Avoid `for` loops over batch dimensions. Use `torch` or `numpy` vectorization.
- **Numerical Stability**:
  - Use `log_softmax` instead of `softmax` followed by `log`.
  - Use `torch.logsumexp` for aggregating probabilities.
  - Add epsilon for division: `x / (y + 1e-8)`.
- **Correctness**: When implementing math from `docs/`, cross-reference the equation numbers in comments.

## 4. Documentation & Research Style

### Markdown Files (`docs/`)
- **Headers**: Use hierarchy `#` > `##` > `###`.
- **Math**: Use LaTeX format.
  - Inline: `$x$`
  - Block:
    ```math
    J(\theta) = \mathbb{E}[R(\tau)]
    ```
- **Filename**: `snake_case.md` (e.g., `variance_proof.md`).

### Knowledge Management
- **New Derivations**: Place in `docs/tasks/taskX.Y/`.
- **References**: When citing a paper, add it to `knowledgebase/papers/` if possible or cite the arXiv ID.

## 5. Workflow Rules for Agents

1. **Read First**: Before editing code, read `docs/PRD.md` and related derivations to understand the *why*.
2. **Test Driven**: If asked to implement a feature (e.g., "Implement the LOO baseline"), create `tests/test_loo.py` *before* or *simultaneously* with the implementation.
3. **No Placeholders**: Do not leave `pass` or `TODO` in critical logic paths unless explicitly instructed.
4. **Verify Environment**: If a tool fails (e.g., `pytest` not found), do not halluncinate output. Ask the user to install dependencies or fall back to inspecting code logic.
5. **Git Hygiene**:
   - Commits should be atomic (one feature/fix per commit).
   - Messages: `Task 1.2: Implement LOO baseline function`.

## 6. Error Handling Strategy

- **Input Validation**: Check tensor shapes and types at public API boundaries.
  ```python
  assert rewards.ndim == 2, "Rewards must be [batch, n_samples]"
  ```
- **Exceptions**: Use specific exceptions (`ValueError`, `RuntimeError`) rather than generic `Exception`.
- **Silent Failures**: Avoid them. If a gradient estimator encounters NaNs, raise an error or log a warning immediately.

---
*This file is the source of truth for agent behavior in this repo. Update it if new tools or patterns are adopted.*
