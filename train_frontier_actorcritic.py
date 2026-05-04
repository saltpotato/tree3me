from __future__ import annotations

import random
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn

from tree_core import Tree, embeds, random_tree_exact_size

try:
    from progress_server import STATUS, start_progress_server
except Exception:
    STATUS = None

    def start_progress_server(*args, **kwargs):
        return None


# -----------------------------
# Version / settings
# -----------------------------

APP_VERSION = "tree3-policy-v0.5-frontier-ac-rollout-signal"

LABEL_COUNT = 3
TREE_SIZE = 6
ATTEMPTS_PER_MOVE = 5
MAX_STEPS = 300

MAX_TREE_TOKENS = 64

D_MODEL = 96
N_HEAD = 4
TREE_ENCODER_LAYERS = 2
D_FF = 192

LR = 5e-5
ENTROPY_BONUS = 0.003
VALUE_LOSS_WEIGHT = 0.5

TRAIN_EPISODES = 5000
PRINT_EVERY = 50
EVAL_EVERY = 250
EVAL_EPISODES = 50

ROLLOUT_BONUS_WEIGHT = 0.05
ROLLOUT_BONUS_COUNT = 2
ROLLOUT_BONUS_MAX_EXTRA_STEPS = 80

MODEL_PATH = "models/policy_model.pt"


VOCAB = {
    "<PAD>": 0,
    "(": 1,
    ")": 2,
    "1": 3,
    "2": 4,
    "3": 5,
}


# -----------------------------
# Tree serialization
# -----------------------------

def tree_to_string(t: Tree) -> str:
    if not t.children:
        return f"({t.label})"

    children = " ".join(tree_to_string(c) for c in t.children)
    return f"({t.label} {children})"


def tokenize_tree(t: Tree) -> torch.Tensor:
    text = tree_to_string(t)
    text = text.replace("(", " ( ").replace(")", " ) ")

    ids: list[int] = []

    for tok in text.split():
        if tok in VOCAB:
            ids.append(VOCAB[tok])

    ids = ids[:MAX_TREE_TOKENS]
    ids += [VOCAB["<PAD>"]] * (MAX_TREE_TOKENS - len(ids))

    return torch.tensor(ids, dtype=torch.long)


def encode_tree_batch(trees: list[Tree], device: str) -> torch.Tensor:
    if not trees:
        return torch.empty((0, MAX_TREE_TOKENS), dtype=torch.long, device=device)

    return torch.stack([tokenize_tree(t) for t in trees]).to(device)


# -----------------------------
# Environment
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

def choose_rollout_policy(valid_candidates: list[Tree]) -> Tree:
    # Keep it simple and stable: mostly heuristic, sometimes random.
    if random.random() < 0.20:
        return random.choice(valid_candidates)

    # Import/use tree_score only for rollout evaluation, not as model input.
    from tree_core import tree_score
    return max(valid_candidates, key=tree_score)


def rollout_after_choice(history_after_choice: list[Tree]) -> int:
    h = list(history_after_choice)
    start_len = len(h)
    max_len = min(MAX_STEPS, start_len + ROLLOUT_BONUS_MAX_EXTRA_STEPS)

    while len(h) < max_len:
        valid = generate_valid_candidates(h)

        if not valid:
            break

        h.append(choose_rollout_policy(valid))

    return len(h) - start_len


def estimate_choice_bonus(history: list[Tree], chosen: Tree) -> float:
    scores = []

    for _ in range(ROLLOUT_BONUS_COUNT):
        h = list(history)
        h.append(chosen)
        scores.append(rollout_after_choice(h))

    best = max(scores) if scores else 0

    # Normalize to roughly 0..1
    return best / max(1, ROLLOUT_BONUS_MAX_EXTRA_STEPS)

# -----------------------------
# Model
# -----------------------------

class TreeEncoder(nn.Module):
    """
    Encodes one tree from raw bracket tokens into one vector.

    No hand features:
    no height
    no label counts
    no candidate_score
    """

    def __init__(self):
        super().__init__()

        self.token_emb = nn.Embedding(len(VOCAB), D_MODEL)
        self.pos_emb = nn.Embedding(MAX_TREE_TOKENS, D_MODEL)

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
            num_layers=TREE_ENCODER_LAYERS,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [batch, MAX_TREE_TOKENS]
        returns: [batch, D_MODEL]
        """
        batch, seq_len = x.shape

        pos = torch.arange(seq_len, device=x.device)
        pos = pos.unsqueeze(0).expand(batch, seq_len)

        h = self.token_emb(x) + self.pos_emb(pos)

        pad_mask = x.eq(VOCAB["<PAD>"])

        h = self.encoder(h, src_key_padding_mask=pad_mask)

        mask = (~pad_mask).float().unsqueeze(-1)
        pooled = (h * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)

        return pooled


class FrontierActorCritic(nn.Module):
    """
    For each candidate:
      - encode candidate tree
      - encode all history trees
      - candidate attends over history tree embeddings
      - output policy logit and value estimate
    """

    def __init__(self):
        super().__init__()

        self.tree_encoder = TreeEncoder()

        self.q = nn.Linear(D_MODEL, D_MODEL)
        self.k = nn.Linear(D_MODEL, D_MODEL)
        self.v = nn.Linear(D_MODEL, D_MODEL)

        self.combine = nn.Sequential(
            nn.LayerNorm(D_MODEL * 4),
            nn.Linear(D_MODEL * 4, D_MODEL),
            nn.GELU(),
            nn.Linear(D_MODEL, D_MODEL),
            nn.GELU(),
        )

        self.policy_head = nn.Linear(D_MODEL, 1)
        self.value_head = nn.Linear(D_MODEL, 1)

    def score_candidates(
        self,
        history_tokens: torch.Tensor,
        candidate_tokens: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        history_tokens:   [H, L]
        candidate_tokens: [K, L]

        returns:
          policy_logits: [K]
          values:        [K]
        """
        candidate_vecs = self.tree_encoder(candidate_tokens)  # [K, D]

        if history_tokens.shape[0] == 0:
            frontier_context = torch.zeros_like(candidate_vecs)
        else:
            history_vecs = self.tree_encoder(history_tokens)  # [H, D]

            q = self.q(candidate_vecs)  # [K, D]
            k = self.k(history_vecs)    # [H, D]
            v = self.v(history_vecs)    # [H, D]

            attn_scores = q @ k.T / (D_MODEL ** 0.5)  # [K, H]
            attn_weights = torch.softmax(attn_scores, dim=1)
            frontier_context = attn_weights @ v       # [K, D]

        x = torch.cat(
            [
                candidate_vecs,
                frontier_context,
                candidate_vecs * frontier_context,
                torch.abs(candidate_vecs - frontier_context),
            ],
            dim=1,
        )

        z = self.combine(x)

        policy_logits = self.policy_head(z).squeeze(-1)
        values = self.value_head(z).squeeze(-1)

        return policy_logits, values


# -----------------------------
# Episode
# -----------------------------

@dataclass
class EpisodeResult:
    length: int
    loss: torch.Tensor | None
    greedy: bool


def run_episode(
    model: FrontierActorCritic,
    optimizer: torch.optim.Optimizer | None,
    device: str,
    greedy: bool = False,
) -> EpisodeResult:
    rewards: list[float] = []
    history: list[Tree] = []

    log_probs: list[torch.Tensor] = []
    entropies: list[torch.Tensor] = []
    chosen_values: list[torch.Tensor] = []

    while len(history) < MAX_STEPS:
        valid_candidates = generate_valid_candidates(history)

        if not valid_candidates:
            break

        history_tokens = encode_tree_batch(history, device)
        candidate_tokens = encode_tree_batch(valid_candidates, device)

        logits, values = model.score_candidates(history_tokens, candidate_tokens)
        probs = torch.softmax(logits, dim=0)

        if greedy:
            choice_index = int(torch.argmax(probs).item())
        else:
            dist = torch.distributions.Categorical(probs)
            choice = dist.sample()
            choice_index = int(choice.item())

            log_probs.append(dist.log_prob(choice))
            entropies.append(dist.entropy())
            chosen_values.append(values[choice_index])

        chosen = valid_candidates[choice_index]

        if greedy:
            history.append(chosen)
        else:
            bonus = estimate_choice_bonus(history, chosen)
            rewards.append(1.0 + ROLLOUT_BONUS_WEIGHT * bonus)
            history.append(chosen)

    length = len(history)

    if greedy:
        return EpisodeResult(length=length, loss=None, greedy=True)

    if not log_probs:
        zero_loss = torch.tensor(0.0, device=device, requires_grad=True)
        return EpisodeResult(length=length, loss=zero_loss, greedy=False)

    # Return-to-go: if episode length is N, first move has N reward left,
    # second has N-1, etc.
    returns_list = []
    running = 0.0

    for r in reversed(rewards):
        running += r
        returns_list.append(running)

    returns_list.reverse()

    returns = torch.tensor(
        returns_list,
        dtype=torch.float32,
        device=device,
    )

    # Normalize returns per episode for stability.
    returns_norm = returns
    if len(returns_norm) > 1:
        returns_norm = (returns_norm - returns_norm.mean()) / (returns_norm.std() + 1e-6)

    log_probs_t = torch.stack(log_probs)
    entropies_t = torch.stack(entropies)
    values_t = torch.stack(chosen_values)

    advantages = returns_norm - values_t

    policy_loss = -(log_probs_t * advantages.detach()).sum()
    value_loss = F.mse_loss(values_t, returns_norm)
    entropy_loss = -ENTROPY_BONUS * entropies_t.sum()

    loss = policy_loss + VALUE_LOSS_WEIGHT * value_loss + entropy_loss

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
    model: FrontierActorCritic,
    device: str,
    episodes: int = EVAL_EPISODES,
) -> dict:
    lengths: list[int] = []

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
# Main
# -----------------------------

def save_model(model: FrontierActorCritic) -> None:
    torch.save(
        {
            "version": APP_VERSION,
            "model_state_dict": model.state_dict(),
            "vocab": VOCAB,
            "settings": {
                "label_count": LABEL_COUNT,
                "tree_size": TREE_SIZE,
                "attempts_per_move": ATTEMPTS_PER_MOVE,
                "max_steps": MAX_STEPS,
                "max_tree_tokens": MAX_TREE_TOKENS,
                "d_model": D_MODEL,
                "n_head": N_HEAD,
                "tree_encoder_layers": TREE_ENCODER_LAYERS,
            },
        },
        MODEL_PATH,
    )


def main() -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"version: {APP_VERSION}")
    print(f"device: {device}")
    print("training learned-frontier actor-critic")
    print("full history forest")
    print("no imitation labels")
    print("no tree_score")
    print("reward = episode length")
    print()

    start_progress_server(host="0.0.0.0", port=80, model_path=MODEL_PATH)

    if STATUS is not None:
        STATUS.update(
            version=APP_VERSION,
            status="training",
            train_episodes=TRAIN_EPISODES,
            model_path=MODEL_PATH,
        )

    model = FrontierActorCritic().to(device)

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

            if STATUS is not None:
                STATUS.update(
                    version=APP_VERSION,
                    status="training",
                    episode=episode,
                    train_episodes=TRAIN_EPISODES,
                    last_length=result.length,
                    avg100=round(avg_recent, 2),
                    last_loss=round(
                        result.loss.item() if result.loss is not None else 0.0,
                        6,
                    ),
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

            save_model(model)

            if STATUS is not None:
                STATUS.update(
                    version=APP_VERSION,
                    status="training",
                    episode=episode,
                    last_eval=eval_stats,
                    model_path=MODEL_PATH,
                )

            print(f"saved {MODEL_PATH}")

    print()
    print("final eval")
    final_eval_stats = evaluate(model, device)

    save_model(model)

    if STATUS is not None:
        STATUS.update(
            version=APP_VERSION,
            status="finished",
            episode=TRAIN_EPISODES,
            train_episodes=TRAIN_EPISODES,
            done=True,
            last_eval=final_eval_stats,
            model_path=MODEL_PATH,
        )

    print(f"saved {MODEL_PATH}")


if __name__ == "__main__":
    main()