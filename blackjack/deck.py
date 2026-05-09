"""Standard-pack blackjack draw pile.

The deck models physical cards by blackjack value, not independent value
probabilities. Cards are removed after drawing. When the draw pile is empty,
only cards that are not currently active on the table are reshuffled.
"""

from __future__ import annotations

from dataclasses import dataclass
import random

from blackjack.cards import VALUES, fresh_value_counts, value_counts


@dataclass(frozen=True)
class DrawResult:
    value: int
    reshuffled: bool


class StandardDeck:
    """A 52-card draw pile represented by blackjack value counts."""

    def __init__(self, rng: random.Random | None = None) -> None:
        self.rng = rng or random.Random()
        self.counts = list(fresh_value_counts())
        self.shuffle_count = 0

    @property
    def remaining(self) -> int:
        return sum(self.counts)

    def reset(self, active_cards: list[int] | tuple[int, ...] = ()) -> None:
        """Reshuffle all non-active cards back into the draw pile.

        Active cards are still physically on the table, including the hidden
        dealer card, so they must remain unavailable after a reshuffle.
        """

        active_counts = value_counts(active_cards)
        self.counts = [
            max(fresh - active, 0)
            for fresh, active in zip(fresh_value_counts(), active_counts)
        ]
        self.shuffle_count += 1

    def draw(self, active_cards: list[int] | tuple[int, ...] = ()) -> DrawResult:
        """Draw one physical card and report whether this draw reshuffled.

        The assignment says the draw pile is reshuffled only after running out
        of cards. If that happens mid-round, the reshuffle excludes cards
        already active on the table.
        """

        reshuffled = False
        if self.remaining == 0:
            self.reset(active_cards)
            reshuffled = True

        pick = self.rng.randrange(self.remaining)
        cumulative = 0
        for index, count in enumerate(self.counts):
            cumulative += count
            if pick < cumulative:
                self.counts[index] -= 1
                return DrawResult(value=VALUES[index], reshuffled=reshuffled)

        raise RuntimeError("Failed to draw from a non-empty deck")


# Backwards-compatible alias for older imports/checkpoints around this module.
OneSuitDeck = StandardDeck
