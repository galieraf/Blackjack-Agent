"""Use a trained DQN model for a one-shot physical blackjack recommendation.

This script does not simulate a round. It converts manually entered table
information into the same state vector used during training, then asks the
trained model for the best currently legal action. Enter current visible
table cards in their own prompts. "Previously seen cards since reshuffle"
means cards from completed earlier rounds after the latest shuffle, not cards
already listed as your hand, dealer upcard, or visible opponent cards.

For real physical play, ``scripts.live_session`` is preferred because it
remembers the card history and current table state across commands.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from agent.dqn import DQNAgent
from blackjack.actions import ACTION_NAMES, DOUBLE, HIT, STAND
from blackjack.cards import parse_cards, parse_card
from blackjack.env import build_live_state


def parse_args() -> argparse.Namespace:
    """Read command-line live-play settings."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=Path("models/dqn_blackjack.pt"))
    parser.add_argument("--num-players", type=int, default=5)
    return parser.parse_args()


def yes_no(raw: str) -> bool:
    """Parse a permissive yes/no answer for the first-action question."""

    return raw.strip().lower() in {"y", "yes", "true", "1"}


def main() -> None:
    """Ask for visible table information and print the recommended action."""

    args = parse_args()
    agent = DQNAgent.load(str(args.checkpoint))

    player_cards = parse_cards(input("Your cards (e.g. A 7): "))
    opponent_cards = parse_cards(input("Visible opponent cards, if any: "))
    dealer_upcard = parse_card(input("Dealer visible card: "))
    previous_seen = parse_cards(input("Previously seen cards since reshuffle, if any: "))
    can_double = yes_no(input("Is this your first action? [y/N]: "))

    state = build_live_state(
        player_cards=player_cards,
        dealer_upcard=dealer_upcard,
        visible_opponent_cards=opponent_cards,
        seen_cards_since_shuffle=previous_seen,
        can_double=can_double,
        num_players=args.num_players,
    )
    legal_actions = (HIT, STAND, DOUBLE) if can_double else (HIT, STAND)
    action = agent.select_action(state, legal_actions, epsilon=0.0)
    q_values = agent.q_values(state)

    print(f"Recommended action: {ACTION_NAMES[action]}")
    print("Q-values:")
    for index, value in enumerate(q_values):
        legality = "legal" if index in legal_actions else "illegal"
        print(f"  {ACTION_NAMES[index]}: {value:.3f} ({legality})")


if __name__ == "__main__":
    main()
