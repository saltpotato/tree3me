from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Tuple
import random

@dataclass(frozen=True)
class Tree:
    label: int
    children: Tuple["Tree", ...] = ()

    @property
    def size(self) -> int:
        return 1 + sum(c.size for c in self.children)

    def pretty(self, indent: str = "") -> str:
        lines = [f"{indent}{self.label}"]
        for child in self.children:
            lines.append(child.pretty(indent + "  "))
        return "\n".join(lines)

def embeds(a: Tree, b: Tree) -> bool:
    """
    TREE/Kruskal-style unordered homeomorphic embedding.

    a embeds into b if:
    - a can match at b's root, or
    - a embeds somewhere inside one of b's children.

    Label condition:
        a.label <= b.label

    Children are unordered:
        children of a must be matched injectively into children/descendant
        structure of b.
    """

    @lru_cache(maxsize=None)
    def rec(x: Tree, y: Tree) -> bool:
        # x may embed directly at y
        if x.label <= y.label and children_embed(x.children, y.children):
            return True

        # or x may embed deeper inside y
        return any(rec(x, child_y) for child_y in y.children)

    @lru_cache(maxsize=None)
    def children_embed(xs: Tuple[Tree, ...], ys: Tuple[Tree, ...]) -> bool:
        """
        Can every child subtree in xs be embedded into a distinct child subtree of ys?

        Since children are unordered, this is a bipartite matching problem.
        This simple recursive version is fine for small experiments.
        """
        if not xs:
            return True

        if len(xs) > len(ys):
            return False

        first_x = xs[0]
        rest_xs = xs[1:]

        for j, yj in enumerate(ys):
            if rec(first_x, yj):
                remaining_ys = ys[:j] + ys[j + 1:]
                if children_embed(rest_xs, remaining_ys):
                    return True

        return False

    return rec(a, b)


def contains_label(t: Tree, label: int) -> bool:
    return t.label == label or any(contains_label(c, label) for c in t.children)


def count_label(t: Tree, label: int) -> int:
    return (1 if t.label == label else 0) + sum(count_label(c, label) for c in t.children)


def height(t: Tree) -> int:
    if not t.children:
        return 1
    return 1 + max(height(c) for c in t.children)


def leaf_count(t: Tree) -> int:
    if not t.children:
        return 1
    return sum(leaf_count(c) for c in t.children)


def max_branching(t: Tree) -> int:
    return max([len(t.children)] + [max_branching(c) for c in t.children])


def total_branching(t: Tree) -> int:
    return len(t.children) + sum(total_branching(c) for c in t.children)


def branching_penalty(t: Tree) -> int:
    return total_branching(t)


def tree_features(t: Tree) -> tuple[int, ...]:
    """
    Simple numeric feature vector for later ML.
    No ML yet. Just stable hand-made features.
    """
    return (
        t.size,
        height(t),
        leaf_count(t),
        max_branching(t),
        total_branching(t),
        t.label,
        count_label(t, 1),
        count_label(t, 2),
        count_label(t, 3),
    )


def history_features(history: list[Tree]) -> tuple[int, ...]:
    """
    Compact features of the current accepted sequence.
    Useful later as ML state input.
    """
    if not history:
        return (0, 0, 0, 0, 0, 0)

    sizes = [t.size for t in history]
    heights = [height(t) for t in history]

    return (
        len(history),
        min(sizes),
        max(sizes),
        sum(sizes),
        max(heights),
        sum(count_label(t, 1) for t in history),
    )


def tree_score(t: Tree) -> int:
    """
    Current heuristic baseline.
    Later ML should try to beat this.
    """
    return (
        - count_label(t, 1) * 200
        - branching_penalty(t) * 5
        + height(t) * 20
        + count_label(t, 3) * 10
    )

def propose_candidate(history, min_size: int, max_size: int, label_count: int) -> Tree:
    """
    As the sequence gets longer, drift toward larger trees.
    """
    growth_bonus = len(history) // 5

    size = random.randint(
        min_size + growth_bonus,
        max_size + growth_bonus,
    )

    return random_tree_exact_size(
        size=size,
        label_count=label_count,
        avoid_label_1=(len(history) < 40),
        max_children_limit=4,
    )

def random_composition_positive(total: int, parts: int) -> tuple[int, ...]:
    """
    Split total into `parts` positive integers.
    Example: total=10, parts=3 -> (2,5,3)
    """
    if parts == 1:
        return (total,)

    cuts = sorted(random.sample(range(1, total), parts - 1))
    values = []
    prev = 0

    for cut in cuts:
        values.append(cut - prev)
        prev = cut

    values.append(total - prev)
    return tuple(values)


def random_tree_exact_size(
    size: int,
    label_count: int = 3,
    avoid_label_1: bool = True,
    max_children_limit: int = 4,
) -> Tree:
    """
    Builds one random tree with exactly `size` nodes.
    Does not enumerate all possible trees.
    """

    if size < 1:
        raise ValueError("size must be >= 1")

    if avoid_label_1:
        labels = list(range(2, label_count + 1))  # [2, 3]
    else:
        labels = list(range(1, label_count + 1))  # [1, 2, 3]

    label = random.choice(labels)

    if size == 1:
        return Tree(label)

    remaining = size - 1

    max_children = min(remaining, max_children_limit)
    child_count = random.randint(1, max_children)

    parts = random_composition_positive(remaining, child_count)

    children = tuple(
        random_tree_exact_size(
            size=part,
            label_count=label_count,
            avoid_label_1=avoid_label_1,
            max_children_limit=max_children_limit,
        )
        for part in parts
    )

    return Tree(label, children)

def verify_history(history: list[Tree]) -> bool:
    """
    Safety check: verifies the whole bad-sequence condition.
    """
    for i in range(len(history)):
        for j in range(i + 1, len(history)):
            if embeds(history[i], history[j]):
                print("BAD SEQUENCE VIOLATION")
                print("i =", i + 1)
                print(history[i].pretty())
                print("j =", j + 1)
                print(history[j].pretty())
                return False

    return True
