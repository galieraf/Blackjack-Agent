"""DQN implementation for the blackjack assignment.

The agent is a standard Deep Q-Network: it learns action values from replayed
transitions and uses a target network for more stable bootstrapped targets.
The blackjack-specific part is the legal-action mask, which prevents double
from being chosen or valued after it is no longer allowed.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import random
from typing import Iterable

try:
    import torch
    from torch import nn
    import torch.nn.functional as F
except ModuleNotFoundError as exc:  # pragma: no cover - exercised when dependency is missing
    torch = None
    nn = None
    F = None
    _TORCH_IMPORT_ERROR = exc
else:
    _TORCH_IMPORT_ERROR = None


def require_torch() -> None:
    if torch is None:
        raise RuntimeError("PyTorch is required. Install dependencies with: pip install -r requirements.txt") from _TORCH_IMPORT_ERROR


@dataclass(frozen=True)
class Transition:
    """One replay-memory item used by Q-learning."""

    state: tuple[float, ...]
    action: int
    reward: float
    next_state: tuple[float, ...]
    done: bool
    legal_next_actions: tuple[int, ...]


class ReplayBuffer:
    """Fixed-size random replay memory for decorrelating training samples."""

    def __init__(self, capacity: int, rng: random.Random | None = None) -> None:
        self.memory: deque[Transition] = deque(maxlen=capacity)
        self.rng = rng or random.Random()

    def __len__(self) -> int:
        return len(self.memory)

    def push(self, transition: Transition) -> None:
        self.memory.append(transition)

    def sample(self, batch_size: int) -> list[Transition]:
        return self.rng.sample(list(self.memory), batch_size)


class QNetwork(nn.Module if nn is not None else object):
    """Small fully connected network that maps a state to three Q-values."""

    def __init__(self, state_size: int, action_size: int = 3, hidden_sizes: tuple[int, ...] = (128, 128)) -> None:
        require_torch()
        super().__init__()
        layers: list[nn.Module] = []
        input_size = state_size
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(input_size, hidden_size))
            layers.append(nn.ReLU())
            input_size = hidden_size
        layers.append(nn.Linear(input_size, action_size))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


class DQNAgent:
    """Deep Q-learning agent with epsilon-greedy legal action selection."""

    def __init__(
        self,
        state_size: int,
        action_size: int = 3,
        hidden_sizes: tuple[int, ...] = (128, 128),
        lr: float = 1e-3,
        gamma: float = 1.0,
        buffer_size: int = 50_000,
        batch_size: int = 128,
        seed: int | None = None,
        device: str | None = None,
    ) -> None:
        require_torch()
        self.state_size = state_size
        self.action_size = action_size
        self.hidden_sizes = hidden_sizes
        self.gamma = gamma
        self.batch_size = batch_size
        self.rng = random.Random(seed)
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.online = QNetwork(state_size, action_size, hidden_sizes).to(self.device)
        self.target = QNetwork(state_size, action_size, hidden_sizes).to(self.device)
        self.target.load_state_dict(self.online.state_dict())
        self.target.eval()
        self.optimizer = torch.optim.Adam(self.online.parameters(), lr=lr)
        self.replay = ReplayBuffer(buffer_size, self.rng)

    def select_action(
        self,
        state: tuple[float, ...],
        legal_actions: Iterable[int],
        epsilon: float = 0.0,
    ) -> int:
        """Choose an epsilon-greedy action from the legal action set only."""

        legal = tuple(legal_actions)
        if not legal:
            raise ValueError("No legal actions available")
        if self.rng.random() < epsilon:
            return self.rng.choice(legal)

        with torch.no_grad():
            state_tensor = torch.tensor([state], dtype=torch.float32, device=self.device)
            q_values = self.online(state_tensor)[0]
            # Illegal actions get -inf before argmax, so a high Q-value for
            # an unavailable action such as late double cannot be selected.
            mask = torch.full((self.action_size,), float("-inf"), device=self.device)
            mask[list(legal)] = 0.0
            return int(torch.argmax(q_values + mask).item())

    def train_step(self) -> float | None:
        """Run one DQN update from a random replay batch.

        The target uses only legal actions in the next state. Terminal states
        have no future value, so their bootstrap term is forced to zero.
        """

        if len(self.replay) < self.batch_size:
            return None

        batch = self.replay.sample(self.batch_size)
        states = torch.tensor([t.state for t in batch], dtype=torch.float32, device=self.device)
        actions = torch.tensor([t.action for t in batch], dtype=torch.int64, device=self.device).unsqueeze(1)
        rewards = torch.tensor([t.reward for t in batch], dtype=torch.float32, device=self.device)
        next_states = torch.tensor([t.next_state for t in batch], dtype=torch.float32, device=self.device)
        dones = torch.tensor([t.done for t in batch], dtype=torch.bool, device=self.device)

        current_q = self.online(states).gather(1, actions).squeeze(1)
        with torch.no_grad():
            next_q = self.target(next_states)
            masks = torch.full((len(batch), self.action_size), float("-inf"), device=self.device)
            for row, transition in enumerate(batch):
                if transition.legal_next_actions:
                    masks[row, list(transition.legal_next_actions)] = 0.0
            max_next_q = torch.max(next_q + masks, dim=1).values
            max_next_q = torch.where(dones, torch.zeros_like(max_next_q), max_next_q)
            target_q = rewards + self.gamma * max_next_q

        loss = F.smooth_l1_loss(current_q, target_q)
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.online.parameters(), 10.0)
        self.optimizer.step()
        return float(loss.item())

    def update_target(self) -> None:
        self.target.load_state_dict(self.online.state_dict())

    def q_values(self, state: tuple[float, ...]) -> list[float]:
        with torch.no_grad():
            tensor = torch.tensor([state], dtype=torch.float32, device=self.device)
            return [float(value) for value in self.online(tensor)[0].cpu().tolist()]

    def save(self, path: str) -> None:
        torch.save(
            {
                "state_size": self.state_size,
                "action_size": self.action_size,
                "hidden_sizes": self.hidden_sizes,
                "model_state_dict": self.online.state_dict(),
            },
            path,
        )

    @classmethod
    def load(cls, path: str, device: str | None = None) -> "DQNAgent":
        require_torch()
        checkpoint = torch.load(path, map_location=device or "cpu")
        agent = cls(
            state_size=checkpoint["state_size"],
            action_size=checkpoint.get("action_size", 3),
            hidden_sizes=tuple(checkpoint.get("hidden_sizes", (128, 128))),
            device=device,
        )
        agent.online.load_state_dict(checkpoint["model_state_dict"])
        agent.update_target()
        return agent
