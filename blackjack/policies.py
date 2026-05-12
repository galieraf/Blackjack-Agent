"""Fixed policies used to simulate other players and baselines.

These policies are intentionally simple. They are not the main agent; they
provide predictable behavior for simulated opponents and a quick comparison
point when evaluating the learned policy.
"""

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
    """Pick uniformly from actions the environment says are currently legal."""

    return rng.choice(legal_actions)


def basic_training_policy(cards: list[int], can_double: bool) -> int:
    """A simple non-learning baseline used for smoke comparisons.

    It doubles only on common strong starting totals and otherwise follows a
    conservative hit/stand rule. This keeps baseline behavior easy to explain.
    """

    total = hand_value(cards)
    soft = usable_ace(cards)
    if can_double and total in (10, 11):
        return DOUBLE
    if soft and total <= 17:
        return HIT
    return HIT if total < 16 else STAND


def basic_strategy_policy(cards: list[int], dealer_upcard: int, can_double: bool) -> int:
    """Compact full-deck basic strategy for hit/stand/double blackjack.

    This is not used as the final agent by default; it gives DQN a stronger
    bootstrap policy because it conditions decisions on the dealer upcard.
    """

    total = hand_value(cards)
    soft = usable_ace(cards)

    if soft:
        if total >= 19:
            return STAND
        if total == 18:
            if can_double and 3 <= dealer_upcard <= 6:
                return DOUBLE
            if dealer_upcard in (2, 7, 8):
                return STAND
            return HIT
        if can_double:
            if total in (13, 14) and dealer_upcard in (5, 6):
                return DOUBLE
            if total in (15, 16) and 4 <= dealer_upcard <= 6:
                return DOUBLE
            if total == 17 and 3 <= dealer_upcard <= 6:
                return DOUBLE
        return HIT

    if total >= 17:
        return STAND
    if 13 <= total <= 16:
        return STAND if 2 <= dealer_upcard <= 6 else HIT
    if total == 12:
        return STAND if 4 <= dealer_upcard <= 6 else HIT
    if total == 11:
        return DOUBLE if can_double and dealer_upcard != 1 else HIT
    if total == 10:
        return DOUBLE if can_double and 2 <= dealer_upcard <= 9 else HIT
    if total == 9:
        return DOUBLE if can_double and 3 <= dealer_upcard <= 6 else HIT
    return HIT
