# Experiment: RL Algorithms in a Snake Game Environment

Adapted from the Patrick Loeber's tutorial. For more detail: https://github.com/patrickloeber/snake-ai-pytorch

## Overview
This project experiments with classic value-based Reinforcement Learning (RL) algorithms on the Snake game. The focus is on:

- **Q-Learning (off-policy)**
- **SARSA (on-policy)**

It provides a clean training loop, reproducible configs, and utilities to compare learning curves and policies across both algorithms.

## Features
- Discrete action space (e.g., `left`, `straight`, `right`).
- Tabular or function-approximation backends (e.g., PyTorch `nn.Module`) for Q-values.
- ε-greedy exploration with decay scheduling.
- Logging of returns, episode lengths, epsilon, and loss.
- Optional replay buffer for Q-Learning with function approximation.
- Config-driven experiments for easy switching between algorithms.

## Installation
```bash
# clone your repo first, then
conda create -n snakerl_env python=3.12
conda activate snakerl_env
pip install -r requirements.txt
```

## Environment API (minimal)
```python
# snake_env.py
class SnakeEnv:
    def reset(self):
        """Returns initial state (numpy array or torch tensor)."""
    def step(self, action):
        """Returns (next_state, reward, done, info)."""
    def render(self):
        """Optional human render using pygame."""
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

## ε-Greedy with Decay (example)
```python
def decay_epsilon(eps, rate=0.995, min_eps=0.05):
    return max(min_eps, eps * rate)
```

Common schedules: exponential (above), linear, or stepwise. Consider tying decay to episode index or average reward stability.

## Training Loop (sketch)
```python
def train(env, agent, episodes=1000, max_steps=500):
    stats = []
    for ep in range(episodes):
        s = env.reset()
        a = agent.act(s) if agent.name == "sarsa" else None
        total_r = 0
        for t in range(max_steps):
            if agent.name == "sarsa":
                s_next, r, done, _ = env.step(a)
                a_next = agent.act(s_next)
                agent.update(s, a, r, s_next, a_next, done)
                s, a = s_next, a_next
            else:  # Q-Learning
                a = agent.act(s)
                s_next, r, done, _ = env.step(a)
                agent.update(s, a, r, s_next, done)
                s = s_next

            total_r += r
            if done:
                break
        agent.epsilon = decay_epsilon(agent.epsilon)
        stats.append({"episode": ep, "return": total_r, "epsilon": agent.epsilon, "steps": t+1})
    return stats
```

## Evaluation
- Run greedy policy (ε=0.0) over N episodes and report mean/median return.
- Optional: save a GIF/MP4 of a rollout.

## Logging & Plots
- Log to CSV per episode: return, steps, epsilon, losses.
- Plot learning curves: moving average of return vs. episodes; epsilon vs. episodes.

## Tips for Stability
- Normalize/clip rewards if needed.
- For function approximation: use target networks or lower learning rates.
- Keep state representation compact and Markovian (avoid redundant features).
- Cap episode length to prevent rare, ultra-long games from destabilizing learning.

## Troubleshooting
- **Diverging Q-values**: lower α, add gradient clipping, check reward signs.
- **No learning**: verify state features, reward shaping, and action mapping.
- **Too random**: ensure ε decays; verify greedy action selection code path.

## Roadmap
- Double Q-Learning baseline
- Expected SARSA
- Prioritized replay (for function approximation)
- Dueling networks (if using deep Q)
- Unit tests for env dynamics & agents

## How to Run
```bash
# Q-Learning
python -m training.train --algo qlearning --episodes 2000 --render 0

# SARSA
python -m training.train --algo sarsa --episodes 2000 --render 0

# Evaluate
python -m training.evaluate --checkpoint checkpoints/qlearning_last.pt --episodes 50 --render 1
```

## Acknowledgments
- Adapted from the FreeCodeCamp Snake RL tutorial. This repo restructures and extends the tutorial for reproducible experiments and side-by-side algorithm comparisons.

## License
Choose a license (e.g., MIT) and include it in the repository root.
