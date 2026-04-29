from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from itertools import product
from typing import Iterable, Tuple



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


def integer_compositions(n: int) -> Iterable[Tuple[int, ...]]:
    """
    Ordered compositions of n.
    Example: n=3 -> (3), (1,2), (2,1), (1,1,1)
    Used to split remaining nodes among children.
    """
    if n == 0:
        yield ()
        return

    for first in range(1, n + 1):
        for rest in integer_compositions(n - first):
            yield (first,) + rest


@lru_cache(maxsize=None)
def shapes_of_size(size: int) -> Tuple[Tuple, ...]:
    """
    Generate unlabeled ordered tree shapes.

    Shape format:
        ()                      leaf
        (child_shape_1, ...)    internal node
    """
    if size < 1:
        return ()

    if size == 1:
        return ((),)

    result = []

    remaining = size - 1

    for child_sizes in integer_compositions(remaining):
        child_shape_lists = [shapes_of_size(s) for s in child_sizes]

        for children in product(*child_shape_lists):
            result.append(children)

    return tuple(result)


def label_shape(shape: Tuple, labels: range) -> Iterable[Tree]:
    """
    Assign labels to every node of an unlabeled shape.
    """
    child_variants = [list(label_shape(child, labels)) for child in shape]

    for root_label in labels:
        if not child_variants:
            yield Tree(root_label)
        else:
            for children in product(*child_variants):
                yield Tree(root_label, tuple(children))


def trees_of_size(size: int, label_count: int = 3) -> Iterable[Tree]:
    """
    Generate all ordered labeled trees with exactly `size` nodes.
    Labels are 1..label_count.
    """
    labels = range(1, label_count + 1)

    for shape in shapes_of_size(size):
        yield from label_shape(shape, labels)


def all_trees(label_count: int = 3) -> Iterable[Tree]:
    """
    Infinite generator:
    size 1, then size 2, then size 3, ...
    """
    size = 1

    while True:
        yield from trees_of_size(size, label_count)
        size += 1
from functools import lru_cache
from dataclasses import dataclass
from typing import Tuple


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

import random


def valid_next(history, candidate):
    """
    TREE bad-sequence condition:
    no earlier tree may embed into the later candidate.
    """
    return all(not embeds(old, candidate) for old in history)


def random_tree_of_size(size: int, label_count: int = 3) -> Tree:
    candidates = list(trees_of_size(size, label_count))
    return random.choice(candidates)


def random_tree_between_sizes(min_size: int, max_size: int, label_count: int = 3) -> Tree:
    size = random.randint(min_size, max_size)
    return random_tree_of_size(size, label_count)

def is_not_too_destructive(t: Tree) -> bool:
    # Toy heuristic: avoid root label 1 early.
    return t.label != 1

def random_tree_all_label_3(size: int) -> Tree:
    shape = random.choice(list(shapes_of_size(size)))

    def build(s):
        return Tree(3, tuple(build(c) for c in s))

    return build(shape)

def acceptable_candidate(history, candidate):
    # Do not allow tiny poison trees early.
    if len(history) < 40 and candidate.size < 4:
        return False

    # Avoid label 1 early.
    if len(history) < 25 and contains_label(candidate, 1):
        return False

    return valid_next(history, candidate)


def contains_label(t, label):
    return t.label == label or any(contains_label(c, label) for c in t.children)

def count_label(t, label):
    return (1 if t.label == label else 0) + sum(count_label(c, label) for c in t.children)


def tree_score(t):
    return (
        t.size * 100
        - count_label(t, 1) * 80
        - count_label(t, 2) * 10
    )

def count_label(t, label):
    return (1 if t.label == label else 0) + sum(count_label(c, label) for c in t.children)

if __name__ == "__main__":
    import random

    random.seed()

    label_count = 3
    target_steps = 50

    # Less destructive start than 3 -> 3
    history = [Tree(3, (Tree(2),))]

    print("accepted 1")
    print(history[0].pretty())

    min_size = 2
    max_size = 6
    attempts_per_step = 1000

    test = Tree(2, (Tree(2),))
    print("test candidate:")
    print(test.pretty())
    print("valid?", valid_next(history, test))

    while len(history) < target_steps:
        valid_candidates = []

        for attempt in range(1, attempts_per_step + 1):
            candidate = random_tree_between_sizes(
                min_size=min_size,
                max_size=max_size,
                label_count=label_count,
            )

            if acceptable_candidate(history, candidate):
                valid_candidates.append(candidate)

            if attempt % 1000 == 0:
                print(
                    f"step {len(history) + 1}, "
                    f"attempt {attempt}, "
                    f"max_size {max_size}, "
                    f"valid {len(valid_candidates)}"
                )

            # stop sampling early once we have enough choices
            if len(valid_candidates) >= 100:
                break

        if valid_candidates:
            # Toy heuristic: tiny trees are usually too general, so choose the largest.
            candidate = max(valid_candidates, key=tree_score)

            history.append(candidate)

            print()
            print(f"accepted {len(history)}")
            print(f"size = {candidate.size}")
            print(candidate.pretty())
        else:
            print()
            print(f"failed after {attempts_per_step} attempts at step {len(history) + 1}")
            print(f"increasing max_size from {max_size} to {max_size + 2}")
            max_size += 2

    print()
    print(f"done: built {len(history)} trees")