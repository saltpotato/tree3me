# TREE3 Self-Play Policy Experiment

Small experimental ML project for learning policies in a TREE/Kruskal-style bad-sequence game.

The goal is **not** to compute TREE(3).  
The goal is to train a model that chooses good next trees in a constrained tree-sequence environment.

## Game idea

We build a sequence of finite labeled trees:

```text
T1, T2, T3, ...
````

A candidate tree is legal only if no previous tree embeds into it:

```python
for old in history:
    assert not embeds(old, candidate)
```

The exact embedding rule is implemented in `tree_core.py`.

The agent receives legal candidates and chooses one.
Reward is simple:

```text
+1 for every accepted tree
episode ends when no legal candidate is found
```

## Current main track

The current main experiment is **self-play policy learning**:

```text
model chooses candidate
environment checks legality with embeds()
episode length becomes reward
policy is updated from its own results
```

No heuristic teacher is required.

## Files

```text
tree_core.py
    Tree class
    embedding checker
    random tree generator

train_policy_selfplay.py
    structural token policy model
    self-play training loop
    saves model checkpoint

models/
    saved model checkpoints

outputs/
    logs or future run outputs

archive_track_a/
    old imitation-learning experiments
```

## Run locally

```bash
python train_policy_selfplay.py
```

If CUDA is available, PyTorch should use the GPU automatically.

## Docker

Build:

```bash
docker build -t tree3-policy .
```

Run:

```bash
docker run --rm tree3-policy
```

With GPU:

```bash
docker run --rm --gpus all tree3-policy
```

## Notes

This is a toy research environment.

The exact TREE/Kruskal embedding checker remains hand-coded because it defines the game rules.
The learning model is responsible for strategy: choosing which legal candidate keeps the future open longest.