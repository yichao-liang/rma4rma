# RMA²: Rapid Motor Adaptation for Robotic Manipulator Arms

[![arXiv](https://img.shields.io/badge/arXiv-2312.04670-b31b1b.svg)](https://arxiv.org/abs/2312.04670)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)

Reference implementation of **"Rapid Motor Adaptation for Robotic Manipulator Arms"**
(Liang, Ellis, Henriques, 2024). RMA² extends Rapid Motor Adaptation — originally
developed for legged locomotion and in-hand rotation — to general object manipulation
with a 7-DOF robot arm, using low-resolution depth perception and category/instance
dictionaries as a proxy for geometry-aware control.

## Method

Training proceeds in two phases (see Fig. 2 of the paper):

1. **Policy training.** A PPO policy is conditioned on privileged environment
   information (object dimensions, density, friction, goal, identity embedding).
   An environment encoder `μ(e, s_t)` distills this into a low-dimensional
   embedding `z_t` that conditions the policy.
2. **Adapter training.** The policy and encoder are frozen. A CNN + temporal
   conv adapter `φ(x_{≤t}, a_{≤t}, f_t)` is trained by L² regression to predict
   `z_t` from proprioception history and a depth image `d_t` — signals available
   at deployment. Category and instance identity are implicitly recovered from
   the depth stream.

At deployment, only the trained CNN `ψ*`, adapter `φ*`, and policy `π*` are used.

## Tasks

Four ManiSkill2 tasks, each with randomized object scale, density, friction,
shape, external disturbance force, and observation noise:

| Task                    | Env id              | Object set             |
| ----------------------- | ------------------- | ---------------------- |
| Pick and Place (YCB)    | `PickSingleYCB-v1`  | 78 YCB objects         |
| Pick and Place (EGAD)   | `PickSingleEGAD-v1` | 2281 EGAD objects      |
| Peg Insertion           | `PegInsertionSide-v1` | cuboid peg (3mm clearance) |
| Faucet Turning          | `TurnFaucet-v1`     | 60 PartNet-Mobility faucets |

See Table 1 of the paper for headline results.

## Installation

The codebase depends on two customized forks of external projects, included as
git submodules:

- [`ManiSkill2`](https://github.com/yichao-liang/ManiSkill2) on branch `rma2`
- [`stable-baselines3`](https://github.com/yichao-liang/stable-baselines3) on branch `rma2`

Clone with submodules:

```bash
git clone --recurse-submodules https://github.com/yichao-liang/rma4rma
cd rma4rma
```

Create a conda environment and install both forks and this package:

```bash
conda env create -f environment.yml
conda activate rma2
pip install -e ".[develop]"
```

> **Note.** Training requires a CUDA GPU; evaluation figures in the paper were
> produced on a single Nvidia A100.

## Usage

All commands below assume the package is installed (`pip install -e .`) and
the `rma2` conda environment is active. They can be run via the console entry
point `rma4rma-train` or as `python -m rma4rma.train`.

### 1. Policy (base) training

```bash
python -m rma4rma.train \
    -e PickSingleYCB-v1 \
    -n 50 -bs 5000 -rs 2000 \
    --randomized_training --ext_disturbance --obs_noise
```

### 2. Adapter training

Requires a base-policy checkpoint (default: `best_model.zip`):

```bash
python -m rma4rma.train \
    -e PickSingleYCB-v1 \
    -n 50 -bs 5000 -rs 2000 \
    --randomized_training --ext_disturbance --obs_noise \
    --adaptation_training --use_depth_adaptation \
    --log_name PPO-ps-bs5000-rs2000-kl0.05-neps10-cr0.2-lr_scdl0-cr_scdl1-ms50-incObsN0_1 \
    --ckpt_name best_model.zip
```

### 3. Evaluation

```bash
python -m rma4rma.train --eval \
    -e PickSingleYCB-v1 \
    --randomized_training --ext_disturbance --obs_noise \
    --use_depth_adaptation \
    --log_name PPO-ps-bs5000-rs2000-kl0.05-neps10-cr0.2-lr_scdl0-cr_scdl1-ms50-incObsN0_1_stage2_dep_1 \
    --ckpt_name best_model.zip
```

### Ablations / baselines

The same entry point supports the ablations reported in Table 1 via flags:

| Variant                    | Flag(s)                                  |
| -------------------------- | ---------------------------------------- |
| Oracle (privileged access) | `--expert_adapt`                         |
| Domain Randomization (DR)  | `--only_DR`                              |
| DR + Vision (DR+Vi)        | `--only_DR --use_depth_base`             |
| Automatic DR (ADR)         | `--auto_dr`                              |
| Without Object Embedding   | `--inc_obs_noise_in_priv` off + no dict  |
| No Vision in Adaptation    | `--adaptation_training` without `--use_depth_adaptation` |

## Repository layout

```
src/rma4rma/
├── algo/                 # RL algorithm: PPO, adapter, policy, buffers
│   ├── adaptation.py     # Adapter training loop
│   ├── buffer.py         # Dict rollout buffer with proprio history
│   ├── callbacks.py      # Checkpoint / eval callbacks
│   ├── evaluate_policy.py
│   ├── misc.py           # Env wrappers, schedules, utilities
│   ├── models.py         # FeaturesExtractor, AdaptationNet, DepthCNN
│   ├── policy.py         # ActorCriticPolicy with privileged/adapter branches
│   └── ppo.py            # PPO subclass with ADR support
├── tasks/                # ManiSkill2 task subclasses with randomization
│   ├── peg_insertion.py
│   ├── pick_cube.py
│   ├── pick_single.py
│   ├── stack_cube.py
│   └── turn_faucet.py
├── config.py             # Argparse + env/log config
└── train.py              # Entry point
tests/                    # Smoke tests (non-GPU)
```

## Development

```bash
./run_autoformat.sh   # black + isort + docformatter
./run_ci_checks.sh    # mypy + pylint + pytest
```

## Citation

```bibtex
@article{liang2024rma,
  title   = {Rapid Motor Adaptation for Robotic Manipulator Arms},
  author  = {Liang, Yichao and Ellis, Kevin and Henriques, Jo{\~a}o},
  journal = {arXiv preprint arXiv:2312.04670},
  year    = {2024}
}
```

## Acknowledgements

This work was supported by the Royal Academy of Engineering (RF\201819\18\163)
and the Cambridge Trust. We thank Doug Morrison for permission to reuse the
EGAD dataset figures, and the authors of ManiSkill2, stable-baselines3, and
RMA for their open-source work that this project builds on.
