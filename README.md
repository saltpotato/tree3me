# TREE(3) Bad Sequence Toy Explorer

This is a small experimental Python project for exploring TREE/Kruskal-style bad sequences of finite labeled rooted trees.

The goal is **not** to compute TREE(3).
The goal is to build intuition for the dynamics behind TREE-like sequences:

- tree generation
- homeomorphic embedding
- forbidden future states
- random/semi-random survival walks
- how small accepted trees can destroy the future

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

The experiment tries to construct such a sequence by randomly proposing candidate trees and accepting only those that satisfy the embedding restriction.

## TREE-Style Embedding

The embedding relation used here is intended to model the Kruskal/TREE intuition:

- finite rooted trees
- node labels from `{1, 2, 3}`
- label order: `1 <= 2 <= 3`
- a node labeled `1` may embed into `1`, `2`, or `3`
- a node labeled `2` may embed into `2` or `3`
- a node labeled `3` may embed only into `3`
- ancestor/descendant structure must be preserved
- intermediate nodes in the target tree may be skipped
- children are treated as unordered or ordered depending on the implementation variant

A valid next tree must satisfy:

```python
def valid_next(history, candidate):
    return all(not embeds(old, candidate) for old in history)
```

## Why This Is Hard

A tree can be valid and still be strategically terrible.

For example, accepting a tiny tree like:

```text
3
```

is often catastrophic because it embeds into every future tree that contains any `3`.

Likewise, trees containing label `1` are dangerous because `1` embeds into everything above it in the label order.

This creates a useful toy version of the TREE phenomenon:

> A locally valid move can destroy huge regions of the future search space.

## Current Strategy

The program does not enumerate all trees of large size.
Instead, it directly generates random trees with a requested node count.

Candidate generation roughly follows this flow:

1. Generate a random candidate.
2. Check whether old trees embed into it.
3. If valid, add it to a candidate pool.
4. Score the valid candidates.
5. Accept the best-scoring one.

The score currently favors:

- larger trees
- fewer label-1 nodes
- fewer overly general structures
- less destructive candidate shapes

This is only a heuristic.

## Running

```bash
python tree.py
```

Example output:

```text
accepted 1
size = 20
3
2
3
2

step 2, attempt 1000, valid 83, size-range 20-30

accepted 2
size = 28
...
```

## Important Warning

This project does not compute TREE(3).

TREE(3) is vastly beyond brute-force computation.

This project is a toy simulation for studying small fragments of the behavior:

- how the forbidden frontier grows
- how bad sequences collapse
- why naive random search fails
- why strategic tree choice matters

## Useful Concepts

### Forbidden Frontier

After accepting trees:

```text
T1, T2, ..., Tn
```

the forbidden region is:

```text
all trees X where some Ti embeds into X
```

So every accepted tree removes a whole upward-closed region of the future tree space.

The program is effectively trying to survive in the remaining space.

### Collapse Phase

A run often starts with large, specific trees.

Later it may accidentally accept smaller, more general trees such as:

```text
2
3
```

or:

```text
1
1
```

These trees block many future candidates.

Eventually the program may reach:

```text
valid 0
```

meaning that no valid candidate was found in the current sampled size range.

This is a toy version of the search space freezing.

## Project Structure

Current file layout:

```text
tree3/
+-- tree.py
+-- render_tree3_sequence.py
+-- tree_log.txt
+-- tree3_qhd.png
+-- result.txt
`-- README.md
```

Suggested future structure:

```text
tree3/
+-- tree_model.py      # Tree dataclass and pretty printing
+-- generator.py       # random tree generation
+-- embedding.py       # Kruskal/TREE embedding oracle
+-- search.py          # candidate search and scoring
+-- main.py
+-- requirements.txt
`-- README.md
```

## Future Improvements

Possible next steps:

- add a full history verifier
- visualize accepted trees
- export sequences to JSON
- compare different scoring functions
- implement unordered-tree canonicalization
- implement a faster embedding checker
- track forbidden-frontier statistics
- add a simple reinforcement learning environment
- create an animation of the bad-sequence walk

## Requirements

For now, a short `requirements.txt` could be:

```txt
numpy
tqdm
```

You only need more later if you add machine learning or visualization.

## Philosophy

This experiment treats TREE not as a number to calculate, but as a survival game in a shrinking space of possible structures.

Each accepted tree is both creation and destruction:

- it extends the sequence
- but it also forbids a huge class of future trees

The central question becomes:

> How do you choose a tree that is valid now, but does not destroy the future?
