"""One-suit blackjack deck."""

from __future__ import annotations

from dataclasses import dataclass
import random

from blackjack.cards import VALUES, fresh_value_counts


@dataclass(frozen=True)
class DrawResult:
    value: int
    reshuffled: bool


class OneSuitDeck:
    """A 13-card one-suit deck, drawn without replacement until empty."""

    def __init__(self, rng: random.Random | None = None) -> None:
        self.rng = rng or random.Random()
        self.counts = list(fresh_value_counts())
        self.shuffle_count = 0

    @property
    def remaining(self) -> int:
        return sum(self.counts)

    def reset(self) -> None:
        self.counts = list(fresh_value_counts())
        self.shuffle_count += 1

    def draw(self) -> DrawResult:
        reshuffled = False
        if self.remaining == 0:
            self.reset()
            reshuffled = True

        pick = self.rng.randrange(self.remaining)
        cumulative = 0
        for index, count in enumerate(self.counts):
            cumulative += count
            if pick < cumulative:
                self.counts[index] -= 1
                return DrawResult(value=VALUES[index], reshuffled=reshuffled)

        raise RuntimeError("Failed to draw from a non-empty deck")
