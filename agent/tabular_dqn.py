"""Tabular Q-learning agent for the blackjack assignment.

The update is the same one-step Q-learning target used by DQN, but action
values are stored directly in a dictionary keyed by a compact discrete state.
This gives a small, explainable baseline that is useful for comparing against
the neural-network DQN.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import random
from typing import Iterable

from blackjack.cards import FRESH_VALUE_COUNTS


@dataclass(frozen=True)
class TabularTransition:
    """One transition consumed by the tabular Q-learning update."""

    state: tuple[float, ...]
    action: int
    reward: float
    next_state: tuple[float, ...]
    done: bool
    legal_next_actions: tuple[int, ...]


class TabularDQNAgent:
    """Epsilon-greedy tabular action-value agent with legal-action masking."""

    def __init__(
        self,
        action_size: int = 3,
        alpha: float = 0.08,
        gamma: float = 1.0,
        seed: int | None = None,
    ) -> None:
        self.action_size = action_size
        self.alpha = alpha
        self.gamma = gamma
        self.rng = random.Random(seed)
        self.q_table: dict[tuple[int, ...], list[float]] = {}

    def select_action(
        self,
        state: tuple[float, ...],
        legal_actions: Iterable[int],
        epsilon: float = 0.0,
    ) -> int:
        """Choose an epsilon-greedy action from legal actions only."""

        legal = tuple(legal_actions)
        if not legal:
            raise ValueError("No legal actions available")
        if self.rng.random() < epsilon:
            return self.rng.choice(legal)

        values = self._values_for_state(state)
        best_value = max(values[action] for action in legal)
        best_actions = [action for action in legal if values[action] == best_value]
        return self.rng.choice(best_actions)

    def update(self, transition: TabularTransition) -> float:
        """Apply one masked Q-learning update and return the TD error."""

        key = self.state_key(transition.state)
        values = self._values_for_key(key)
        current = values[transition.action]
        if transition.done or not transition.legal_next_actions:
            next_value = 0.0
        else:
            next_values = self._values_for_state(transition.next_state)
            next_value = max(next_values[action] for action in transition.legal_next_actions)
        target = transition.reward + self.gamma * next_value
        td_error = target - current
        values[transition.action] = current + self.alpha * td_error
        return td_error

    def q_values(self, state: tuple[float, ...]) -> list[float]:
        """Return a copy of the action values for display/evaluation."""

        return list(self._values_for_state(state))

    def _values_for_state(self, state: tuple[float, ...]) -> list[float]:
        return self._values_for_key(self.state_key(state))

    def _values_for_key(self, key: tuple[int, ...]) -> list[float]:
        if key not in self.q_table:
            self.q_table[key] = [0.0 for _ in range(self.action_size)]
        return self.q_table[key]

    @staticmethod
    def state_key(state: tuple[float, ...]) -> tuple[int, ...]:
        """Discretize the environment state into an explainable table key."""

        if len(state) >= 71:
            player_total = max(0, min(31, round(state[0] * 31)))
            soft = 1 if state[1] >= 0.5 else 0
            can_double = 1 if state[2] >= 0.5 else 0
            dealer_features = state[36:46]
            dealer_upcard = max(range(10), key=lambda index: dealer_features[index]) + 1
            remaining_slice = state[46:56]
            opponent_card_bin = max(0, min(20, round(state[68] * 20)))
            opponents = max(0, min(4, round(state[69] * 4)))
        else:
            player_total = max(0, min(31, round(state[0] * 31)))
            soft = 1 if state[1] >= 0.5 else 0
            dealer_upcard = max(1, min(10, round(state[2] * 10)))
            can_double = 1 if state[3] >= 0.5 else 0
            remaining_slice = state[4:14]
            visible_opponent_cards = 0
            for feature, fresh in zip(state[14:24], FRESH_VALUE_COUNTS):
                visible_opponent_cards += round(feature * fresh)
            opponent_card_bin = max(0, min(20, visible_opponent_cards))
            opponents = max(0, min(4, round(state[24] * 4)))

        remaining_counts = [
            max(0, min(fresh, round(feature * fresh)))
            for feature, fresh in zip(remaining_slice, FRESH_VALUE_COUNTS)
        ]
        seen_counts = [fresh - remaining for fresh, remaining in zip(FRESH_VALUE_COUNTS, remaining_counts)]
        hi_lo = (
            sum(seen_counts[1:6])
            - seen_counts[0]
            - seen_counts[9]
        )
        count_bin = max(-6, min(6, hi_lo))

        return (
            player_total,
            soft,
            dealer_upcard,
            can_double,
            count_bin,
            opponent_card_bin,
            opponents,
        )

    def save(self, path: str | Path) -> None:
        """Write a portable JSON checkpoint."""

        data = {
            "action_size": self.action_size,
            "alpha": self.alpha,
            "gamma": self.gamma,
            "q_table": {
                ",".join(str(part) for part in key): values
                for key, values in self.q_table.items()
            },
        }
        Path(path).write_text(json.dumps(data), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "TabularDQNAgent":
        """Load a JSON checkpoint produced by ``save``."""

        data = json.loads(Path(path).read_text(encoding="utf-8"))
        agent = cls(
            action_size=data.get("action_size", 3),
            alpha=data.get("alpha", 0.08),
            gamma=data.get("gamma", 1.0),
        )
        agent.q_table = {
            tuple(int(part) for part in key.split(",")): [float(value) for value in values]
            for key, values in data.get("q_table", {}).items()
        }
        return agent
