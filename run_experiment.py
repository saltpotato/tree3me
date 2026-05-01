import random
from statistics import mean
import csv
import os

from tree_core import (
    Tree,
    history_features,
    random_tree_exact_size,
    tree_features,
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

def choose_heuristic_epsilon(valid_candidates: list[Tree]) -> Tree:
    if random.random() < 0.10:
        return random.choice(valid_candidates)
    return max(valid_candidates, key=tree_score)

def append_training_rows(csv_path, rows):
    file_exists = os.path.exists(csv_path)

    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "episode_id",
                "agent",
                "seed",
                "step",
                "candidate_index",
                "chosen",
                "episode_final_length",

                "history_len",
                "history_min_size",
                "history_max_size",
                "history_sum_size",
                "history_max_height",
                "history_label1_count",

                "candidate_size",
                "candidate_height",
                "candidate_leaf_count",
                "candidate_max_branching",
                "candidate_total_branching",
                "candidate_root_label",
                "candidate_label1_count",
                "candidate_label2_count",
                "candidate_label3_count",
                "candidate_score",
                "remaining_after_choice",
                "capped",
            ],
        )

        if not file_exists:
            writer.writeheader()

        writer.writerows(rows)

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
    collect_data: bool = False,
    episode_id: int = 0,
    agent_name: str = "",
) -> list[Tree] | tuple[list[Tree], list[dict]]:
    
    if seed is not None:
        random.seed(seed)

    history: list[Tree] = []
    data_rows = []
    pending_rows = []

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

            if collect_data:
                h_feat = history_features(history)

            pending_rows = []

            for idx, cand in enumerate(valid_candidates):
                c_feat = tree_features(cand)

                pending_rows.append({
                    "episode_id": episode_id,
                    "agent": agent_name,
                    "seed": seed,
                    "step": len(history) + 1,
                    "candidate_index": idx,
                    "chosen": 1 if cand == chosen else 0,
                    "episode_final_length": -1,  # filled after episode ends

                    "history_len": h_feat[0],
                    "history_min_size": h_feat[1],
                    "history_max_size": h_feat[2],
                    "history_sum_size": h_feat[3],
                    "history_max_height": h_feat[4],
                    "history_label1_count": h_feat[5],

                    "candidate_size": c_feat[0],
                    "candidate_height": c_feat[1],
                    "candidate_leaf_count": c_feat[2],
                    "candidate_max_branching": c_feat[3],
                    "candidate_total_branching": c_feat[4],
                    "candidate_root_label": c_feat[5],
                    "candidate_label1_count": c_feat[6],
                    "candidate_label2_count": c_feat[7],
                    "candidate_label3_count": c_feat[8],
                    "candidate_score": tree_score(cand),
                    "remaining_after_choice": -1,
                    "capped": -1,
                })

            data_rows.extend(pending_rows)

            history.append(chosen)

        if verbose:
            print()
            print(f"accepted {len(history)}")
            print(f"size = {chosen.size}")
            print(chosen.pretty())

        if verify:
            assert verify_history(history)

    if collect_data:
        final_length = len(history)
        capped = 1 if final_length >= max_steps else 0

        for row in data_rows:
            row["episode_final_length"] = final_length
            row["remaining_after_choice"] = final_length - row["step"]
            row["capped"] = capped

        return history, data_rows

    return history

from concurrent.futures import ProcessPoolExecutor

AGENTS = {
    # "random": choose_random,
    # "largest": choose_largest,
    # "heuristic": choose_heuristic,
    "heuristic_epsilon": choose_heuristic_epsilon,
}

def run_episode_job(args):
    name, seed, max_steps = args

    result = run_benchmark_episode(
        chooser=AGENTS[name],
        seed=seed,
        label_count=3,
        min_size=6,
        max_size=6,
        attempts_per_move=5,
        max_steps=max_steps,
        verbose=False,
        verify=False,
        collect_data=True,
        episode_id=seed,
        agent_name=name,
    )

    history, rows = result
    
    return len(history), rows

def run_benchmark(
    episodes: int,
    base_seed: int = 1234,
):
    max_steps = 300

    for name in AGENTS:
        scores = []

        jobs = [(name, base_seed + i, max_steps) for i in range(episodes)]

        with ProcessPoolExecutor(max_workers=6) as pool:
            results = list(pool.map(run_episode_job, jobs))

            scores = [score for score, _rows in results]

            all_rows = []
            for _score, rows in results:
                all_rows.extend(rows)

            append_training_rows("training_data.csv", all_rows)

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