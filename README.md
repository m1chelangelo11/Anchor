# Anchor

A self-supervised predictive path-consistency mechanism for latent world models. Anchor encourages consistency between direct, horizon-conditioned multi-step prediction and compositional autoregressive rollout — reducing horizon-dependent compound error accumulation (**latent reality drift**: the divergence between free-running latent rollouts and encoded future observations).

Anchor is not a standalone architecture or merely an auxiliary loss. It is a training-time regularization for shaping stable latent predictive trajectories.

> A model should not only predict the future locally. Different predictive paths leading to the same future should remain mutually consistent and grounded in observed reality.

## Status

Proof of concept on a synthetic, deterministic chaotic environment (Hénon map), used to validate that:
1. autoregressive rollout error accumulates over time in a chaotic system (confirmed)
2. a one-step predictor trained on this environment reproduces the same accumulation (confirmed baseline)
3. the Anchor mechanism reduces this drift compared to rollout-only prediction (in progress)

Current baseline (no Anchor): rollout drift saturates around step 20-25, see `results/`.

## Project structure

```
Anchor/
├── src/
│   ├── toystateworld.py   # environment: Hénon map, rollout, drift metrics, dataset generation
│   ├── model.py            # Predictor network (one-step dynamics F)
│   ├── train.py             # training loop for F
│   └── evaluate.py          # autoregressive rollout evaluation, drift measurement, plots
├── models/                  # saved checkpoints (gitignored, reproducible via seed=42)
├── results/                 # saved plots and drift data
├── requirements.txt
└── README.md
```

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Usage

```bash
python src/train.py      # trains the one-step predictor F, saves to models/predictor.pt
python src/evaluate.py   # runs autoregressive rollout, measures drift, saves plot to results/
```

Both scripts seed numpy and torch (`seed=42`) for reproducibility.

## Roadmap

- [x] Environment: ToyStateWorld (Hénon map)
- [x] Autoregressive rollout dynamics (one-step predictor `F`)
- [x] Baseline: rollout-only, no Anchor
- [ ] Direct horizon head `G_k`
- [ ] Stop-gradient teacher target
- [ ] Anchor loss