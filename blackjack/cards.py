"""Card parsing, deck-count utilities, and blackjack hand scoring.

The project stores cards by blackjack value instead of by suit/rank object.
That is enough for this assignment because the deck is exactly one suit:
Ace, 2-9, and four physical cards worth 10 (10, J, Q, K).
"""

from __future__ import annotations

from collections import Counter

ACE = 1
TEN_VALUE = 10
VALUES = tuple(range(1, 11))
# Counts for values A, 2, ..., 10 in a fresh one-suit deck.
# The final 4 represents the separate physical cards 10, J, Q, and K.
FRESH_VALUE_COUNTS = (1, 1, 1, 1, 1, 1, 1, 1, 1, 4)

CARD_ALIASES = {
    "A": ACE,
    "ACE": ACE,
    "1": ACE,
    "J": TEN_VALUE,
    "JACK": TEN_VALUE,
    "Q": TEN_VALUE,
    "QUEEN": TEN_VALUE,
    "K": TEN_VALUE,
    "KING": TEN_VALUE,
}


def parse_card(raw: str) -> int:
    """Parse a card label into its blackjack value.

    Face cards are represented as value 10. Aces are represented as 1, with
    soft-hand handling done by ``hand_value``.
    """

    token = raw.strip().upper()
    if token in CARD_ALIASES:
        return CARD_ALIASES[token]
    try:
        value = int(token)
    except ValueError as exc:
        raise ValueError(f"Unknown card label: {raw!r}") from exc
    if 2 <= value <= 10:
        return value
    raise ValueError(f"Card value must be A, 2-10, J, Q, or K: {raw!r}")


def parse_cards(raw: str) -> list[int]:
    """Parse comma- or space-separated card labels."""

    if not raw.strip():
        return []
    normalized = raw.replace(",", " ")
    return [parse_card(part) for part in normalized.split()]


def card_label(value: int) -> str:
    if value == ACE:
        return "A"
    if value == TEN_VALUE:
        return "10"
    if 2 <= value <= 9:
        return str(value)
    raise ValueError(f"Invalid card value: {value}")


def hand_value(cards: list[int] | tuple[int, ...]) -> int:
    """Return the best non-busting blackjack total for a hand.

    Aces are stored as 1 first. Each ace can add 10 more points, which makes
    it count as 11, but only while doing so keeps the hand at 21 or below.
    """

    total = sum(cards)
    aces = sum(1 for card in cards if card == ACE)
    while aces and total + 10 <= 21:
        total += 10
        aces -= 1
    return total


def usable_ace(cards: list[int] | tuple[int, ...]) -> bool:
    """Return whether at least one ace is currently counted as 11."""

    total = sum(cards)
    return any(card == ACE for card in cards) and total + 10 <= 21


def is_bust(cards: list[int] | tuple[int, ...]) -> bool:
    return hand_value(cards) > 21


def value_counts(cards: list[int] | tuple[int, ...]) -> tuple[int, ...]:
    counts = Counter(cards)
    return tuple(counts[value] for value in VALUES)


def fresh_value_counts() -> tuple[int, ...]:
    return FRESH_VALUE_COUNTS


def known_remaining_counts(known_cards: list[int] | tuple[int, ...]) -> tuple[int, ...]:
    """Estimate remaining one-suit deck counts from visible known cards.

    The live game can only know cards that have been revealed. Counts are
    clipped at zero so accidental duplicate manual input cannot produce
    negative state features.
    """

    seen = value_counts(known_cards)
    return tuple(max(total - used, 0) for total, used in zip(FRESH_VALUE_COUNTS, seen))
