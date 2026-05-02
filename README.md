
# TREE(3) Bad Sequence Toy Explorer

This is a small experimental Python project for exploring TREE/Kruskal-style bad sequences of finite labeled rooted trees.

The goal is **not** to compute TREE(3). The goal is to build intuition for TREE-like dynamics and to experiment with machine learning agents that try to choose good next trees in a shrinking combinatorial search space.

The project currently explores:
- rooted labeled tree generation
- TREE/Kruskal-style homeomorphic embedding  
- bad-sequence construction
- forbidden future states
- random and heuristic survival policies
- benchmark episodes
- data logging
- imitation learning from a hand-written heuristic

---

## Idea

A sequence of trees
```text
T1, T2, T3, ...
```

is considered valid if no earlier tree embeds into a later tree:

```python
for i < j:
    assert not embeds(T[i], T[j])
```

This is the TREE/Kruskal-style bad sequence condition.

The experiment tries to build long valid sequences by repeatedly proposing candidate trees and accepting only candidates that satisfy the embedding restriction.

⸻

## TREE-Style Embedding

The embedding relation used here models a Kruskal/TREE-like intuition:

* finite rooted trees
* node labels from {1, 2, 3}
* label order: 1 <= 2 <= 3
* a node labeled 1 may embed into 1, 2, or 3
* a node labeled 2 may embed into 2 or 3
* a node labeled 3 may embed only into 3
* ancestor/descendant structure must be preserved
* intermediate nodes in the target tree may be skipped
* children are treated as unordered in the embedding checker

A valid next tree must satisfy:

```python
def valid_next(history, candidate):
    return all(not embeds(old, candidate) for old in history)
```

This means that the new candidate must not contain any previous tree as an embedded structural pattern.

⸻

## Why This Is Hard

A tree can be valid and still be strategically terrible.

For example, accepting a tiny tree like:
```
3
```
is often catastrophic because it embeds into every future tree that contains any 3.

Likewise, trees containing label 1 are dangerous because 1 embeds into everything above it in the label order.

This creates a useful toy version of the TREE phenomenon:

A locally valid move can destroy huge regions of the future search space.

The core challenge is not only to find a valid next tree, but to find one that keeps the future open.

⸻

## Forbidden Frontier

After accepting trees:
```
T1, T2, ..., Tn
```

the forbidden region is:
```
all trees X where some Ti embeds into X
```

Every accepted tree removes an upward-closed region from the future tree space.

The program is effectively trying to survive in the remaining space.

A useful mental model:

Each accepted tree extends the sequence, but also deforms the forbidden frontier.

⸻

## Current Project Structure

Current simplified layout:

```
tree3/
├── tree_core.py
├── run_experiment.py
├── train_imitation_model.py
├── requirements.txt
├── run_pipeline.bat
├── README.md
└── .gitignore
```

Generated files may appear during experiments:
```
training_data.csv
imitation_model.joblib
__pycache__/
```

These are generated artifacts and do not need to be committed.

⸻

## Main Files

### tree_core.py

Contains the core symbolic logic:

* Tree dataclass
* pretty-printing
* TREE/Kruskal-style embeds(a, b)
* random tree generation
* tree feature extraction
* history feature extraction
* heuristic scoring
* optional sequence verification

### run_experiment.py

Runs benchmark and data collection episodes.

Current agents:
* random
* largest  
* heuristic
* heuristic_epsilon
* imitation

The benchmark repeatedly:

1. samples candidate trees
2. filters valid candidates using the exact embedding oracle
3. lets an agent choose one valid candidate
4. appends the chosen tree to history
5. stops when no valid candidates are found or max_steps is reached

### train_imitation_model.py

Trains a RandomForestClassifier to imitate the hand-written heuristic chooser.

The model learns from rows like:
```
history features + candidate features -> chosen / not chosen
```

It saves:
```
imitation_model.joblib
```

### run_pipeline.bat

Runs the full simple pipeline:
```
delete old training_data.csv
collect heuristic training data  
train imitation model
benchmark all agents
```

⸻

## Running

Install requirements:
```bash
pip install -r requirements.txt
```

Or inside the virtual environment:
```bash
myenv\Scripts\activate
python -m pip install -r requirements.txt
```

Collect clean heuristic training data:
```bash
python run_experiment.py --mode collect --episodes 500
```

Train imitation model:
```bash
python train_imitation_model.py
```

Benchmark all agents:
```bash
python run_experiment.py --mode benchmark --episodes 50
```

Or run the pipeline:
```bash
run_pipeline.bat
```

⸻

## Requirements

Current requirements.txt:
```
pandas
scikit-learn
joblib
tqdm
```

Note:
* The pip package is called scikit-learn.
* The Python import is still sklearn.

Example:
```python
from sklearn.ensemble import RandomForestClassifier
```

⸻

## Current Benchmark Setup

The current useful benchmark uses approximately:
```
labels = 3
candidate size = exactly 6
attempts_per_move = 3
max_steps = 300
episodes = 50
```

Each move samples only a few candidate trees. This makes the problem nontrivial because the agent must choose well from a small candidate pool.

A representative benchmark result:
```
random             avg ~44
largest            avg ~45
heuristic          avg ~92
heuristic_epsilon  avg ~83
imitation          avg ~88
```

Interpretation:
* random and largest are weak under fixed-size candidates.
* heuristic is currently the strongest hand-written baseline.
* imitation learns a substantial part of the heuristic behavior and comes close to it.

This shows that the data logging, training, and model-in-the-loop benchmark pipeline works.

⸻

## Imitation Learning

Track A currently uses imitation learning.

The teacher is the hand-written heuristic:
```python
def choose_heuristic(valid_candidates):
    return max(valid_candidates, key=tree_score)
```

Training data records candidate rows:
```
history features
candidate features  
chosen yes/no
```

The imitation model tries to predict which candidate the heuristic would choose.

This is not expected to greatly outperform the heuristic. Its purpose is to prove the ML pipeline:
```
environment -> data collection -> training -> model agent -> benchmark
```

⸻

## Important Limitation

The current imitation model uses handcrafted features such as:
* tree size
* height
* leaf count
* branching
* label counts
* candidate score

This is useful as a first baseline, but it is not the final representation.

A stronger future model should read the tree structure itself, not only handcrafted summaries.

⸻

## Future Direction: Learned Structural Representation

The next serious direction is to let the model learn features from tree structure directly.

Possible approaches:
* serialize trees as bracketed token sequences
* encode structural fragments
* use candidate-conditioned attention over previous tree fragments
* store full history as external memory
* retrieve relevant earlier structures instead of compressing history into one vector

The longer-term idea:

The model should learn which structural elements of the history define the dangerous forbidden frontier for a candidate.

This would move beyond handcrafted feature extraction.

⸻

## Possible Next Steps

Short-term:
* clean up generated files
* keep training_data.csv and imitation_model.joblib out of Git
* stabilize benchmark settings
* add train_value_model.py

Track B:
* train a value model
* target: remaining_after_choice
* use only chosen rows initially
* compare value model against heuristic

Representation-learning track:
* add tree serialization
* log candidate tree strings
* log history tail or history fragments
* train a model that reads structure directly
* later replace handcrafted features

Possible future architecture:
```
candidate tree fragments
+
retrieved history/frontier fragments
+
candidate-conditioned attention
->
candidate value score
```

⸻

## Philosophy

This experiment treats TREE not as a number to calculate, but as a survival game in a shrinking space of possible structures.

Each accepted tree is both creation and destruction:
* it extends the sequence
* but it also forbids a class of future trees

The central question becomes:

How do you choose a tree that is valid now, but does not destroy the future?

The current project is a toy environment for exploring that question with symbolic rules and machine learning.