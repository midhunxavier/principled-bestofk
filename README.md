# Principled Best-of-K RL for Combinatorial Optimization

This repository implements **principled Max@K policy gradients** for Neural Combinatorial Optimization (NCO), providing unbiased gradient estimators as an alternative to heuristic "Leader Reward" approaches.

## Quick Start: Google Colab

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/)

### 1. Setup Cell

Run this cell first to clone the repo and install dependencies:

```python
# Clone the repository
!git clone https://github.com/YOUR_USERNAME/principled-bestofk.git
%cd principled-bestofk

# Install dependencies
!pip install -q torch torchvision torchaudio
!pip install -q rl4co tensordict lightning matplotlib numpy pandas

# Verify installation
!PYTHONPATH=code python3 -c "from src.algorithms.maxk_pomo import MaxKPOMO; print('Setup complete!')"
```

### 2. Run Training with Progress Bar

```python
# Train MaxK POMO on TSP50 with progress bar
!PYTHONPATH=code python3 -m src.experiments.train_tsp \
    --algorithm maxk_pomo \
    --num_loc 50 \
    --num_starts 32 \
    --k 8 \
    --variance_reduction subloo \
    --weight_normalization zscore \
    --min_gap_scale 0.01 \
    --max_epochs 100 \
    --batch_size 128 \
    --train_data_size 20000 \
    --val_data_size 2000 \
    --lr 1e-4 \
    --gradient_clip_val 1.0 \
    --accelerator gpu \
    --devices 1 \
    --log_every_n_steps 50
```

The progress bar will show:
- Current epoch / total epochs
- Training loss
- Validation reward
- Estimated time remaining

### 3. Run Full Experiment Suite

For a complete comparison of all algorithms (POMO, Leader Reward, MaxK POMO):

```python
# Run all algorithms on TSP50 and TSP100
!PYTHONPATH=code python3 -m src.experiments.phase4_tsp_experiments all \
    --runs_dir /content/runs \
    --artifacts_dir /content/artifacts \
    --num_locs 50 100 \
    --seeds 0 1 2 \
    --algorithms pomo leader_reward maxk_pomo \
    --num_starts 32 \
    --maxk_k 8 \
    --maxk_variance_reduction subloo \
    --max_epochs 100 \
    --batch_size 128 \
    --accelerator gpu \
    --devices 1 \
    --k_eval 128
```

---

## Detailed Usage

### Training Individual Algorithms

#### POMO Baseline
```bash
PYTHONPATH=code python3 -m src.experiments.train_tsp \
    --algorithm pomo \
    --num_loc 50 \
    --num_starts 32 \
    --max_epochs 100 \
    --accelerator gpu
```

#### Leader Reward Baseline
```bash
PYTHONPATH=code python3 -m src.experiments.train_tsp \
    --algorithm leader_reward \
    --num_loc 50 \
    --num_starts 32 \
    --alpha 0.5 \
    --max_epochs 100 \
    --accelerator gpu
```

#### MaxK POMO (Ours) with SubLOO
```bash
PYTHONPATH=code python3 -m src.experiments.train_tsp \
    --algorithm maxk_pomo \
    --num_loc 50 \
    --num_starts 32 \
    --k 8 \
    --variance_reduction subloo \
    --weight_normalization zscore \
    --min_gap_scale 0.01 \
    --max_epochs 100 \
    --accelerator gpu
```

#### MaxK POMO with Hybrid Mode (Recommended for Stability)
```bash
PYTHONPATH=code python3 -m src.experiments.train_tsp \
    --algorithm maxk_pomo \
    --num_loc 50 \
    --num_starts 32 \
    --k 8 \
    --variance_reduction hybrid \
    --hybrid_lambda 0.7 \
    --weight_normalization zscore \
    --max_epochs 100 \
    --accelerator gpu
```

### Key Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--num_loc` | 50 | TSP problem size (number of cities) |
| `--num_starts` | 32 | Number of multi-start rollouts (n) |
| `--k` | 8 | Max@K parameter (recommend n/k ratio of 4:1) |
| `--variance_reduction` | `subloo` | Options: `none`, `sample_loo`, `subloo`, `hybrid` |
| `--weight_normalization` | `zscore` | Options: `none`, `zscore`, `sum_to_zero` |
| `--min_gap_scale` | 0.01 | Minimum gap floor for SubLOO (prevents zero gradients) |
| `--hybrid_lambda` | 0.5 | Blend coefficient for hybrid mode (1.0=SubLOO, 0.0=POMO) |
| `--max_epochs` | 100 | Training epochs |
| `--gradient_clip_val` | 1.0 | Gradient clipping (critical for stability) |

---

## Monitoring Training Progress

### Progress Bar
The training script shows a live progress bar by default:

```
Epoch 42/100: 100%|██████████| 156/156 [01:23<00:00, 1.87it/s, loss=0.0234, val/reward=-5.98]
```

### TensorBoard (Optional)
```python
# In Colab, run this before training:
%load_ext tensorboard
%tensorboard --logdir /content/runs
```

### Training Curves
After training, generate visualization:
```python
!PYTHONPATH=code python3 -m src.experiments.phase4_tsp_experiments summarize \
    --runs_dir /content/runs \
    --artifacts_dir /content/artifacts

# View the generated plots
from IPython.display import Image
Image('/content/artifacts/plots/t4_1_tsp50_val_cost.png')
```

---

## Google Colab Complete Notebook

Copy this entire cell block into a Colab notebook:

```python
#@title Setup and Installation
!git clone https://github.com/YOUR_USERNAME/principled-bestofk.git 2>/dev/null || echo "Already cloned"
%cd /content/principled-bestofk

# Install dependencies
!pip install -q torch rl4co tensordict lightning matplotlib tqdm

# Check GPU
import torch
print(f"GPU Available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
```

```python
#@title Training Configuration
ALGORITHM = "maxk_pomo"  #@param ["pomo", "leader_reward", "maxk_pomo"]
NUM_LOC = 50  #@param [20, 50, 100]
NUM_STARTS = 32  #@param [16, 32, 64]
K = 8  #@param [4, 8, 16]
VARIANCE_REDUCTION = "subloo"  #@param ["none", "sample_loo", "subloo", "hybrid"]
WEIGHT_NORMALIZATION = "zscore"  #@param ["none", "zscore", "sum_to_zero"]
HYBRID_LAMBDA = 0.7  #@param {type:"slider", min:0, max:1, step:0.1}
MAX_EPOCHS = 100  #@param {type:"integer"}
BATCH_SIZE = 128  #@param [64, 128, 256]
SEED = 0  #@param {type:"integer"}
```

```python
#@title Run Training with Progress Bar
import subprocess
import sys

cmd = [
    sys.executable, "-m", "src.experiments.train_tsp",
    "--algorithm", ALGORITHM,
    "--num_loc", str(NUM_LOC),
    "--num_starts", str(NUM_STARTS),
    "--k", str(K),
    "--variance_reduction", VARIANCE_REDUCTION,
    "--weight_normalization", WEIGHT_NORMALIZATION,
    "--hybrid_lambda", str(HYBRID_LAMBDA),
    "--max_epochs", str(MAX_EPOCHS),
    "--batch_size", str(BATCH_SIZE),
    "--seed", str(SEED),
    "--gradient_clip_val", "1.0",
    "--accelerator", "gpu" if torch.cuda.is_available() else "cpu",
    "--devices", "1",
    "--log_every_n_steps", "20",
    "--output_dir", "/content/runs",
]

# Run with live output
import os
os.environ["PYTHONPATH"] = "code"
!PYTHONPATH=code {' '.join(cmd)}
```

```python
#@title Visualize Training Curves
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# Find the latest run
runs_dir = Path("/content/runs")
if runs_dir.exists():
    run_dirs = sorted(runs_dir.iterdir(), key=lambda x: x.stat().st_mtime, reverse=True)
    if run_dirs:
        latest_run = run_dirs[0]
        metrics_file = latest_run / "version_0" / "metrics.csv"
        
        if metrics_file.exists():
            df = pd.read_csv(metrics_file)
            
            fig, axes = plt.subplots(1, 2, figsize=(14, 5))
            
            # Plot validation reward (higher is better, so negate for cost)
            val_data = df[df['val/reward'].notna()]
            if not val_data.empty:
                axes[0].plot(val_data['epoch'], -val_data['val/reward'], 'b-', linewidth=2)
                axes[0].set_xlabel('Epoch')
                axes[0].set_ylabel('Validation Cost (lower is better)')
                axes[0].set_title(f'Training Progress: {latest_run.name}')
                axes[0].grid(True, alpha=0.3)
            
            # Plot training loss
            train_data = df[df['train/loss'].notna()]
            if not train_data.empty:
                axes[1].plot(train_data['step'], train_data['train/loss'], 'r-', alpha=0.7)
                axes[1].set_xlabel('Step')
                axes[1].set_ylabel('Training Loss')
                axes[1].set_title('Training Loss')
                axes[1].grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig('/content/training_curves.png', dpi=150)
            plt.show()
            
            # Print final metrics
            final_val = val_data.iloc[-1] if not val_data.empty else None
            if final_val is not None:
                print(f"\nFinal Validation Cost: {-final_val['val/reward']:.4f}")
else:
    print("No runs found. Please run training first.")
```

```python
#@title Evaluate Model (Best-of-K Sampling)
# Find the checkpoint
from pathlib import Path

runs_dir = Path("/content/runs")
run_dirs = sorted(runs_dir.iterdir(), key=lambda x: x.stat().st_mtime, reverse=True)
if run_dirs:
    ckpt = run_dirs[0] / "checkpoints" / "last.ckpt"
    if ckpt.exists():
        print(f"Evaluating: {ckpt}")
        !PYTHONPATH=code python3 -m src.experiments.evaluate \
            --problem tsp \
            --num_loc {NUM_LOC} \
            --algorithm {ALGORITHM} \
            --ckpt_path {str(ckpt)} \
            --method sampling \
            --k_eval 128 \
            --num_instances 1000 \
            --device cuda \
            --save_path /content/eval_results.json
        
        # Display results
        import json
        with open('/content/eval_results.json') as f:
            results = json.load(f)
        print(f"\nBest-of-128 Average Cost: {results['metrics']['avg_cost']:.4f}")
    else:
        print("No checkpoint found")
else:
    print("No runs found")
```

---

## Expected Results

After 100 epochs on TSP50:

| Algorithm | Greedy Cost | Best-of-128 Cost |
|-----------|-------------|------------------|
| POMO | ~6.45 | ~5.98 |
| Leader Reward | ~6.53 | ~6.01 |
| MaxK POMO (SubLOO + zscore) | ~6.48* | ~5.99* |

*With the fixes (weight normalization, min gap floor, n/k=4:1 ratio, 100 epochs)

---

## Troubleshooting

### Out of Memory
Reduce batch size or num_starts:
```bash
--batch_size 64 --num_starts 16
```

### Training Stuck / No Progress
Try hybrid mode for more robust early training:
```bash
--variance_reduction hybrid --hybrid_lambda 0.7
```

### NaN Loss
Enable numeric checks for debugging:
```bash
--check_numerics
```

---

## Citation

If you use this code, please cite:

```bibtex
@article{principled-bestofk,
  title={Principled Best-of-K Policy Gradients for Neural Combinatorial Optimization},
  author={...},
  year={2025}
}
```

## License

MIT License
