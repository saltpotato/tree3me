from __future__ import annotations
from progress_server import STATUS, start_progress_server

import random
from dataclasses import dataclass
from typing import List

import torch
from torch import nn

from tree_core import Tree, embeds, random_tree_exact_size, verify_history

# -----------------------------
# Settings
# -----------------------------
APP_VERSION = "tree3-policy-v0.2-actorcritic-test"

LABEL_COUNT = 3
TREE_SIZE = 6
ATTEMPTS_PER_MOVE = 5
MAX_STEPS = 300
HISTORY_TAIL_SIZE = 6

MAX_LEN = 512
D_MODEL = 96
N_HEAD = 4
N_LAYERS = 3
D_FF = 192

LR = 5e-5
ENTROPY_BONUS = 0.003
TRAIN_EPISODES = 5000
PRINT_EVERY = 50
EVAL_EVERY = 250
EVAL_EPISODES = 50

MODEL_PATH = "models/policy_model.pt"


# -----------------------------
# Raw structural tokenization
# -----------------------------

VOCAB = {
    "<PAD>": 0,
    "(": 1,
    ")": 2,
    "|": 3,
    "H": 4,
    "C": 5,
    "1": 6,
    "2": 7,
    "3": 8,
}


def tree_to_string(t: Tree) -> str:
    if not t.children:
        return f"({t.label})"

    children = " ".join(tree_to_string(c) for c in t.children)
    return f"({t.label} {children})"


def tokenize(text: str) -> list[int]:
    spaced = text.replace("(", " ( ").replace(")", " ) ").replace("|", " | ")

    ids = []
    for tok in spaced.split():
        if tok in VOCAB:
            ids.append(VOCAB[tok])

    return ids


def encode_context(history: list[Tree], candidate: Tree) -> torch.Tensor:
    """
    Raw-ish structure input.

    No hand features.
    No tree_score.
    No height/counts.

    The model sees:
        last K history trees + candidate tree
    serialized as symbolic tree syntax.
    """
    history_tail = history[-HISTORY_TAIL_SIZE:]

    hist_text = " | ".join(tree_to_string(t) for t in history_tail)
    cand_text = tree_to_string(candidate)

    text = f"H {hist_text} C {cand_text}"

    ids = tokenize(text)
    ids = ids[:MAX_LEN]
    ids += [VOCAB["<PAD>"]] * (MAX_LEN - len(ids))

    return torch.tensor(ids, dtype=torch.long)


# -----------------------------
# Environment helpers
# -----------------------------


def generate_valid_candidates(history: list[Tree]) -> list[Tree]:
    candidates: list[Tree] = []

    for _ in range(ATTEMPTS_PER_MOVE):
        cand = random_tree_exact_size(
            size=TREE_SIZE,
            label_count=LABEL_COUNT,
            avoid_label_1=False,
            max_children_limit=4,
        )

        if all(not embeds(old, cand) for old in history):
            candidates.append(cand)

    return candidates


# -----------------------------
# Policy model
# -----------------------------


class StructuralPolicyNet(nn.Module):
    """
    Small structural attention policy.

    It scores one (history_tail + candidate) context.
    During an episode, we score all valid candidates and softmax over them.
    """

    def __init__(self):
        super().__init__()

        self.token_emb = nn.Embedding(len(VOCAB), D_MODEL)
        self.pos_emb = nn.Embedding(MAX_LEN, D_MODEL)

        layer = nn.TransformerEncoderLayer(
            d_model=D_MODEL,
            nhead=N_HEAD,
            dim_feedforward=D_FF,
            dropout=0.1,
            activation="gelu",
            batch_first=True,
        )

        self.encoder = nn.TransformerEncoder(
            layer,
            num_layers=N_LAYERS,
        )

        self.head = nn.Sequential(
            nn.LayerNorm(D_MODEL),
            nn.Linear(D_MODEL, D_MODEL),
            nn.GELU(),
            nn.Linear(D_MODEL, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x shape:
            [batch, MAX_LEN]

        returns:
            [batch]
        """
        batch, seq_len = x.shape

        pos = torch.arange(seq_len, device=x.device)
        pos = pos.unsqueeze(0).expand(batch, seq_len)

        h = self.token_emb(x) + self.pos_emb(pos)

        pad_mask = x.eq(VOCAB["<PAD>"])

        h = self.encoder(h, src_key_padding_mask=pad_mask)

        # mean pooling over non-pad tokens
        mask = (~pad_mask).float().unsqueeze(-1)
        pooled = (h * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)

        return self.head(pooled).squeeze(-1)


# -----------------------------
# One episode
# -----------------------------


@dataclass
class EpisodeResult:
    length: int
    loss: torch.Tensor | None
    greedy: bool


def run_episode(
    model: StructuralPolicyNet,
    optimizer: torch.optim.Optimizer | None,
    device: str,
    greedy: bool = False,
) -> EpisodeResult:
    history: list[Tree] = []

    log_probs: list[torch.Tensor] = []
    entropies: list[torch.Tensor] = []

    while len(history) < MAX_STEPS:
        valid_candidates = generate_valid_candidates(history)

        if not valid_candidates:
            break

        batch = torch.stack(
            [encode_context(history, cand) for cand in valid_candidates]
        ).to(device)

        logits = model(batch)
        probs = torch.softmax(logits, dim=0)

        if greedy:
            choice_index = int(torch.argmax(probs).item())
        else:
            dist = torch.distributions.Categorical(probs)
            choice = dist.sample()
            choice_index = int(choice.item())

            log_probs.append(dist.log_prob(choice))
            entropies.append(dist.entropy())

        chosen = valid_candidates[choice_index]
        history.append(chosen)

    # Optional sanity check for debugging only.
    # Expensive if used constantly.
    # assert verify_history(history)

    length = len(history)

    if greedy:
        return EpisodeResult(length=length, loss=None, greedy=True)

    if not log_probs:
        # No move was made.
        zero_loss = torch.tensor(0.0, device=device, requires_grad=True)
        return EpisodeResult(length=length, loss=zero_loss, greedy=False)

    # REINFORCE:
    # reward is +1 per accepted move.
    # return-to-go at step t is remaining episode length.
    returns = torch.arange(
        len(log_probs),
        0,
        -1,
        dtype=torch.float32,
        device=device,
    )

    # Normalize returns within episode for stability.
    if len(returns) > 1:
        returns = (returns - returns.mean()) / (returns.std() + 1e-6)

    log_probs_t = torch.stack(log_probs)
    entropies_t = torch.stack(entropies)

    policy_loss = -(log_probs_t * returns).sum()
    entropy_loss = -ENTROPY_BONUS * entropies_t.sum()

    loss = policy_loss + entropy_loss

    if optimizer is not None:
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

    return EpisodeResult(length=length, loss=loss.detach(), greedy=False)


# -----------------------------
# Evaluation
# -----------------------------


def evaluate(
    model: StructuralPolicyNet, device: str, episodes: int = EVAL_EPISODES
) -> dict:
    lengths = []

    model.eval()
    with torch.no_grad():
        for _ in range(episodes):
            result = run_episode(
                model=model,
                optimizer=None,
                device=device,
                greedy=True,
            )
            lengths.append(result.length)

    model.train()

    avg = sum(lengths) / len(lengths)

    stats = {
        "episodes": episodes,
        "avg": round(avg, 2),
        "best": max(lengths),
        "worst": min(lengths),
        "lengths": lengths,
    }

    print()
    print("EVAL")
    print(f"episodes: {episodes}")
    print(f"avg length: {avg:.2f}")
    print(f"best: {max(lengths)}")
    print(f"worst: {min(lengths)}")
    print(f"lengths: {lengths}")
    print()

    return stats


# -----------------------------
# Training
# -----------------------------


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("device:", device)
    print("training self-play policy")
    print("no imitation labels")
    print("no tree_score")
    print("reward = episode length")
    print()

    start_progress_server(host="0.0.0.0", port=80, model_path=MODEL_PATH)

    STATUS.update(
        status="training",
        train_episodes=TRAIN_EPISODES,
        model_path=MODEL_PATH,
        version=APP_VERSION,
    )

    model = StructuralPolicyNet().to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LR,
        weight_decay=1e-4,
    )

    recent_lengths: list[int] = []

    for episode in range(1, TRAIN_EPISODES + 1):
        model.train()

        result = run_episode(
            model=model,
            optimizer=optimizer,
            device=device,
            greedy=False,
        )

        recent_lengths.append(result.length)
        if len(recent_lengths) > 100:
            recent_lengths.pop(0)

        if episode % PRINT_EVERY == 0:
            avg_recent = sum(recent_lengths) / len(recent_lengths)

            STATUS.update(
                status="training",
                episode=episode,
                train_episodes=TRAIN_EPISODES,
                last_length=result.length,
                avg100=round(avg_recent, 2),
                last_loss=round(result.loss.item() if result.loss is not None else 0.0, 6),
                model_path=MODEL_PATH,
            )

            print(
                f"episode {episode:5d} | "
                f"length {result.length:4d} | "
                f"avg100 {avg_recent:7.2f} | "
                f"loss {result.loss.item() if result.loss is not None else 0.0:9.4f}"
            )

        if episode % EVAL_EVERY == 0:
            eval_stats = evaluate(model, device)

            STATUS.update(
                status="training",
                episode=episode,
                last_eval=eval_stats,
                model_path=MODEL_PATH,
            )
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "vocab": VOCAB,
                    "max_len": MAX_LEN,
                    "settings": {
                        "label_count": LABEL_COUNT,
                        "tree_size": TREE_SIZE,
                        "attempts_per_move": ATTEMPTS_PER_MOVE,
                        "max_steps": MAX_STEPS,
                        "history_tail_size": HISTORY_TAIL_SIZE,
                    },
                },
                MODEL_PATH,
            )

            print(f"saved {MODEL_PATH}")

    print()
    print("final eval")
    final_eval_stats = evaluate(model, device)

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "vocab": VOCAB,
            "max_len": MAX_LEN,
        },
        MODEL_PATH,
    )

    STATUS.update(
        status="finished",
        episode=TRAIN_EPISODES,
        train_episodes=TRAIN_EPISODES,
        done=True,
        last_eval=final_eval_stats,
        model_path=MODEL_PATH,
    )

    print(f"saved {MODEL_PATH}")


if __name__ == "__main__":
    print(f"version: {APP_VERSION}")
    main()
