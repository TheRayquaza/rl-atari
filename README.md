# RL EPITA FINAL PROJECT - ATARI

mateo.lelong@epita.fr

## Features

- **Hydra Configuration Management**: Easy experimentation with different hyperparameters
- **Multiple Agents**: Q-Learning, SARSA, Q-Learning with epsilon scheduling
- **Automatic Logging**: Training metrics, checkpoints, and visualizations
- **Video Recording**: Automatic video recording during evaluation
- **Grid Search**: Built-in support for hyperparameter tuning
- **Comprehensive Plots**: Automatic generation of training curves and analysis

## Project Structure

```
.
├── configs/
├── main.py                      # Main training script
├── pipeline/                   # Agent pipelines
└── models/                     # Agent implementations
```

## Installation

```bash
uv add -r requirements.txt
uv sync
```

## Lint

```bash
ruff check . --fix
mypy .
```

## Usage

### 1. Basic Training

Train with default configuration:
```bash
python main.py
```

### 2. Override Parameters

Train with custom hyperparameters:
```bash
# Change learning rate and epsilon
python main.py agent.learning_rate=0.2 agent.epsilon=0.05

# Use different agent (specify the group path)
python main.py agent/sarsa=default

# Or use qlearning with decay
python main.py agent/qlearning_decay=default

# Quick training run
python main.py training=quick

# Disable video recording
python main.py evaluation=no_video

# Custom experiment name
python main.py experiment.name=my_experiment
```

### 3. Training Only (No Evaluation)

```bash
python main.py evaluation.enabled=false
```

Or disable video recording only:
```bash
python main.py evaluation=no_video
```

### 4. Evaluation Only (Load Checkpoint)

Modify `main.py` to load a checkpoint:
```python
# In main function, before evaluation
if cfg.evaluation.enabled:
    checkpoint_path = Path("path/to/checkpoint.pkl")
    trainer.load_checkpoint(checkpoint_path)
    trainer.evaluate()
```

## Plotting

The framework automatically generates:

1. **Rewards Plot**: Episode rewards and running average
2. **Episode Lengths**: Steps per episode
3. **Epsilon Decay**: Exploration rate over time
4. **Training Summary**: 4-panel overview
   - Heatmaps of hyperparameter performance
   - Top configurations bar chart
   - Parameter effect plots

## Example Workflow

```bash
# 1. Quick test
python main.py training=quick

# 2. Grid search to find good hyperparameters
python main.py -m \
  agent.learning_rate=0.05,0.1,0.2 \
  agent.epsilon=0.05,0.1,0.2 \
  training=quick

# 3. Analyze results
python grid_search.py

# 4. Full training with best config
python main.py \
  agent.learning_rate=0.1 \
  agent.epsilon=0.1 \
  agent.gamma=0.99 \
  training=default

# 5. Evaluate saved agent
python main.py \
  training.enabled=false \
  evaluation.enabled=true \
  evaluation.episodes=200
```
