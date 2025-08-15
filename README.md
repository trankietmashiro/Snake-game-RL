# Experiment: RL Algorithms in a Snake Game Environment

Adapted from the Patrick Loeber's tutorial. For more detail: https://github.com/patrickloeber/snake-ai-pytorch

## Overview
This project experiments with classic value-based Reinforcement Learning (RL) algorithms on the Snake game. The focus is on:

- **Q-Learning (off-policy)**
- **SARSA (on-policy)**

It provides a training loop, reproducible configs, and utilities to compare learning curves and policies across both algorithms.

## Installation
```bash
# clone your repo first, then
conda create -n snakerl_env python=3.12
conda activate snakerl_env
pip install -r requirements.txt
```

State can be raw grid features or a compact feature vector (e.g., direction, food delta, wall proximity, self-collision flags).

## Algorithms
### Q-Learning (Off-Policy)
Target:
\[ Q(s,a) \leftarrow Q(s,a) + \alpha\big(r + \gamma \max_{a'} Q(s', a') - Q(s,a)\big) \]

### SARSA (On-Policy)
Target:
\[ Q(s,a) \leftarrow Q(s,a) + \alpha\big(r + \gamma Q(s', a') - Q(s,a)\big) \]

Key difference: SARSA uses the next action actually taken under the current policy; Q-Learning uses the greedy action at the next state.

## ε-Greedy with Decay
```python
def decay_epsilon(eps, rate=0.995, min_eps=0.05):
    return max(min_eps, eps * rate)
```

## How to Run
```bash
# Q-Learning
python3 main.py --algo qlearning --episodes 2000

# SARSA
python main.py --algo sarsa --episodes 2000
```

## Acknowledgments
- Adapted from the Patrick Loeber's Snake RL tutorial on freeCodeCamp youtube channel. This repo restructures and extends the tutorial for reproducible experiments and side-by-side algorithm comparisons.
