"""Stateful physical-card live play assistant.

Unlike ``live_play.py``, this script runs as a session. It remembers cards
seen since the latest shuffle, keeps current table cards separate from
completed-round history, and asks the trained DQN for recommendations from
the current visible state.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from pathlib import Path

from agent.dqn import DQNAgent
from blackjack.actions import ACTION_NAMES, DOUBLE, HIT, STAND
from blackjack.cards import card_label, hand_value, known_remaining_counts, parse_card, parse_cards
from blackjack.env import build_live_state


def format_cards(cards: list[int]) -> str:
    """Format stored card values for terminal output."""

    return " ".join(card_label(card) for card in cards) if cards else "-"


@dataclass
class LiveSessionState:
    """Track visible physical-card state during live play.

    ``seen_cards_since_shuffle`` stores cards from completed rounds only.
    Current table cards are kept in the hand fields and are combined with the
    completed-round history when building the model observation.
    """

    num_players: int = 5
    seen_cards_since_shuffle: list[int] = field(default_factory=list)
    my_cards: list[int] = field(default_factory=list)
    dealer_cards: list[int] = field(default_factory=list)
    opponent_cards: list[int] = field(default_factory=list)
    can_double: bool = True

    def shuffle(self) -> str:
        """Forget all known cards because the physical deck was shuffled."""

        self.seen_cards_since_shuffle.clear()
        self.clear_table()
        return "Shuffle recorded."

    def new_round(self) -> str:
        """Start a new round while keeping completed-round card history."""

        self.clear_table()
        return "New round started."

    def clear_table(self) -> None:
        self.my_cards.clear()
        self.dealer_cards.clear()
        self.opponent_cards.clear()
        self.can_double = True

    def deal(self, my_cards: list[int], dealer_upcard: int, opponent_cards: list[int]) -> str:
        """Record initial visible cards for the current round."""

        self.my_cards = list(my_cards)
        self.dealer_cards = [dealer_upcard]
        self.opponent_cards = list(opponent_cards)
        self.can_double = True
        return "Initial deal recorded."

    def me_hit(self, card: int) -> str:
        self.my_cards.append(card)
        self.can_double = False
        return f"My hit recorded: {card_label(card)}."

    def me_double(self, card: int) -> str:
        self.my_cards.append(card)
        self.can_double = False
        return f"My double card recorded: {card_label(card)}."

    def me_stand(self) -> str:
        self.can_double = False
        return "Stand recorded."

    def opp_hit(self, card: int) -> str:
        self.opponent_cards.append(card)
        return f"Opponent hit recorded: {card_label(card)}."

    def opp_cards(self, cards: list[int]) -> str:
        self.opponent_cards.extend(cards)
        return "Opponent cards recorded."

    def dealer_reveal(self, card: int) -> str:
        self.dealer_cards.append(card)
        return f"Dealer reveal recorded: {card_label(card)}."

    def dealer_draw(self, card: int) -> str:
        self.dealer_cards.append(card)
        return f"Dealer draw recorded: {card_label(card)}."

    def round_end(self) -> str:
        """Move current visible table cards into completed-round history."""

        self.seen_cards_since_shuffle.extend(self.current_visible_cards())
        self.clear_table()
        return "Round ended."

    def current_visible_cards(self) -> list[int]:
        return list(self.my_cards) + list(self.dealer_cards) + list(self.opponent_cards)

    def known_cards_for_observation(self) -> list[int]:
        """Return completed-round cards plus current visible table cards."""

        return list(self.seen_cards_since_shuffle) + self.current_visible_cards()

    def legal_actions(self) -> tuple[int, ...]:
        return (HIT, STAND, DOUBLE) if self.can_double else (HIT, STAND)

    def model_state(self) -> tuple[float, ...]:
        """Build the same observation vector used by training."""

        if not self.my_cards:
            raise ValueError("Enter your cards first with: deal")
        if not self.dealer_cards:
            raise ValueError("Enter dealer upcard first with: deal")
        return build_live_state(
            player_cards=self.my_cards,
            dealer_upcard=self.dealer_cards[0],
            visible_opponent_cards=self.opponent_cards,
            seen_cards_since_shuffle=self.seen_cards_since_shuffle,
            can_double=self.can_double,
            num_players=self.num_players,
        )

    def status_text(self) -> str:
        """Return a compact human-readable snapshot of the live state."""

        remaining = known_remaining_counts(self.known_cards_for_observation())
        remaining_text = ", ".join(
            f"{card_label(value)}:{count}" for value, count in zip(range(1, 11), remaining)
        )
        lines = [
            f"My hand: {format_cards(self.my_cards)}"
            + (f" (total {hand_value(self.my_cards)})" if self.my_cards else ""),
            f"Dealer visible: {format_cards(self.dealer_cards)}",
            f"Opponent visible: {format_cards(self.opponent_cards)}",
            f"Completed-round seen cards: {format_cards(self.seen_cards_since_shuffle)}",
            f"Double legal: {'yes' if self.can_double else 'no'}",
            f"Known remaining counts: {remaining_text}",
        ]
        return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=Path("models/dqn_blackjack.pt"))
    parser.add_argument("--num-players", type=int, default=5)
    return parser.parse_args()


def print_help() -> None:
    print(
        "Commands:\n"
        "  shuffle\n"
        "  new round\n"
        "  deal\n"
        "  me hit CARD\n"
        "  me double CARD\n"
        "  me stand\n"
        "  opp hit CARD\n"
        "  opp cards CARD CARD ...\n"
        "  dealer reveal CARD\n"
        "  dealer draw CARD\n"
        "  round end\n"
        "  recommend\n"
        "  status\n"
        "  quit"
    )


def recommend(agent: DQNAgent, state: LiveSessionState) -> str:
    """Return a recommendation and Q-values for the current live state."""

    model_state = state.model_state()
    legal_actions = state.legal_actions()
    action = agent.select_action(model_state, legal_actions, epsilon=0.0)
    q_values = agent.q_values(model_state)
    lines = [f"Recommended action: {ACTION_NAMES[action]}"]
    for index, value in enumerate(q_values):
        legality = "legal" if index in legal_actions else "illegal"
        lines.append(f"  {ACTION_NAMES[index]}: {value:.3f} ({legality})")
    return "\n".join(lines)


def handle_command(command: str, state: LiveSessionState) -> str | None:
    """Apply a non-recommendation command to the session state."""

    parts = command.split()
    if not parts:
        return None
    if command == "shuffle":
        return state.shuffle()
    if command == "new round":
        return state.new_round()
    if command == "deal":
        my_cards = parse_cards(input("Your initial cards: "))
        dealer_upcard = parse_card(input("Dealer upcard: "))
        opponent_cards = parse_cards(input("Opponent visible cards, if any: "))
        return state.deal(my_cards, dealer_upcard, opponent_cards)
    if parts[:2] == ["me", "hit"] and len(parts) == 3:
        return state.me_hit(parse_card(parts[2]))
    if parts[:2] == ["me", "double"] and len(parts) == 3:
        return state.me_double(parse_card(parts[2]))
    if command == "me stand":
        return state.me_stand()
    if parts[:2] == ["opp", "hit"] and len(parts) == 3:
        return state.opp_hit(parse_card(parts[2]))
    if parts[:2] == ["opp", "cards"] and len(parts) > 2:
        return state.opp_cards(parse_cards(" ".join(parts[2:])))
    if parts[:2] == ["dealer", "reveal"] and len(parts) == 3:
        return state.dealer_reveal(parse_card(parts[2]))
    if parts[:2] == ["dealer", "draw"] and len(parts) == 3:
        return state.dealer_draw(parse_card(parts[2]))
    if command == "round end":
        return state.round_end()
    if command == "status":
        return state.status_text()
    if command == "help":
        print_help()
        return None
    raise ValueError(f"Unknown command: {command!r}. Type 'help' for commands.")


def main() -> None:
    args = parse_args()
    agent = DQNAgent.load(str(args.checkpoint))
    state = LiveSessionState(num_players=args.num_players)

    print("Stateful blackjack live session. Type 'help' for commands.")
    while True:
        try:
            command = input("> ").strip().lower()
            if command == "quit":
                break
            if command == "recommend":
                print(recommend(agent, state))
                continue
            result = handle_command(command, state)
            if result:
                print(result)
        except (ValueError, RuntimeError) as exc:
            print(f"Error: {exc}")


if __name__ == "__main__":
    main()
