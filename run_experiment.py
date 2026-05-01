import random
from statistics import mean

from tree_core import (
    Tree,
    random_tree_exact_size,
    tree_score,
    embeds,
    verify_history,
)

def choose_random(valid_candidates: list[Tree]) -> Tree:
    return random.choice(valid_candidates)


def choose_largest(valid_candidates: list[Tree]) -> Tree:
    return max(valid_candidates, key=lambda t: t.size)


def choose_heuristic(valid_candidates: list[Tree]) -> Tree:
    return max(valid_candidates, key=tree_score)


def run_benchmark_episode(
    chooser,
    seed: int | None = None,
    label_count: int = 3,
    min_size: int = 2,
    max_size: int = 10,
    attempts_per_move: int = 100,
    max_steps: int = 50,
    verbose: bool = False,
    verify: bool = True,
) -> list[Tree]:
    if seed is not None:
        random.seed(seed)

    history: list[Tree] = []

    while len(history) < max_steps:
        valid_candidates = []

        for _ in range(attempts_per_move):
            # For benchmark: allow all labels, including 1.
            # This makes the game harder and more honest.

            candidate = random_tree_exact_size(
                size = random.randint(min_size, max_size),
                label_count=label_count,
                avoid_label_1=False,
                max_children_limit=4,
            );

            # For the benchmark, only the real TREE condition counts.
            # No hand filters like "avoid label 1 early".
            if all(not embeds(old, candidate) for old in history):
                valid_candidates.append(candidate)

        if not valid_candidates:
            break

        chosen = chooser(valid_candidates)
        history.append(chosen)

        if verbose:
            print()
            print(f"accepted {len(history)}")
            print(f"size = {chosen.size}")
            print(chosen.pretty())

        if verify:
            assert verify_history(history)

    return history

from concurrent.futures import ProcessPoolExecutor

def run_episode_job(args):
    name, seed, max_steps = args

    chooser_map = {
        "random": choose_random,
        "largest": choose_largest,
        "heuristic": choose_heuristic,
    }

    history = run_benchmark_episode(
        chooser=chooser_map[name],
        seed=seed,
        label_count=3,
        min_size=6,
        max_size=6,
        attempts_per_move=5,
        max_steps=max_steps,
        verbose=False,
        verify=False,
    )

    return len(history)

def run_benchmark(
    episodes: int,
    base_seed: int = 1234,
):
    agents = {
        "random": choose_random,
        "largest": choose_largest,
        "heuristic": choose_heuristic,
    }

    max_steps = 300

    for name, _ in agents.items():
        scores = []

        
        jobs = [(name, base_seed + i) for i in range(episodes)]

        with ProcessPoolExecutor(max_workers=3) as pool:
            scores = list(pool.map(run_episode_job, jobs))

        print()
        print(f"agent: {name}")
        print(f"episodes: {episodes}")
        print(f"avg score: {mean(scores):.2f}")
        print(f"best score: {max(scores)}")
        print(f"worst score: {min(scores)}")
        print(f"scores: {scores}")
        capped = sum(1 for s in scores if s >= max_steps)
        print(f"capped episodes: {capped}/{episodes}")

if __name__ == "__main__":
    run_benchmark(episodes=50)