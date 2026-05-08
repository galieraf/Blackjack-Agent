"""Fixed policies used to simulate other players and baselines."""

from __future__ import annotations

import random

from blackjack.actions import DOUBLE, HIT, STAND
from blackjack.cards import hand_value, usable_ace


def dealer_like_policy(cards: list[int]) -> int:
    """Simple opponent policy: hit below 17, otherwise stand."""

    return HIT if hand_value(cards) < 17 else STAND


def conservative_policy(cards: list[int]) -> int:
    """Baseline policy that avoids drawing on totals above 15."""

    total = hand_value(cards)
    if total <= 11:
        return HIT
    if total <= 15 and usable_ace(cards):
        return HIT
    return HIT if total < 16 else STAND


def random_legal_policy(legal_actions: list[int], rng: random.Random) -> int:
    return rng.choice(legal_actions)


def basic_training_policy(cards: list[int], can_double: bool) -> int:
    """A simple non-learning baseline used for smoke comparisons."""

    total = hand_value(cards)
    soft = usable_ace(cards)
    if can_double and total in (10, 11):
        return DOUBLE
    if soft and total <= 17:
        return HIT
    return HIT if total < 16 else STAND
