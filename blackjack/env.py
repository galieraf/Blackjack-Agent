"""Assignment-specific blackjack environment.

The environment is intentionally small and rule-focused: it simulates the
assignment variant from one player's point of view, returns sparse terminal
rewards, and exposes legal actions so the agent never has to learn invalid
moves by trial and error.
"""

from __future__ import annotations

from dataclasses import dataclass
import random

from blackjack.actions import DOUBLE, HIT, STAND
from blackjack.cards import (
    FRESH_VALUE_COUNTS,
    hand_value,
    is_bust,
    known_remaining_counts,
    usable_ace,
    value_counts,
)
from blackjack.deck import StandardDeck
from blackjack.policies import dealer_like_policy


@dataclass(frozen=True)
class StepResult:
    """Result returned after one environment action.

    ``legal_actions`` belongs to the next state and is stored in replay memory
    so the Q-learning target can ignore actions that would be illegal there.
    """

    state: tuple[float, ...]
    reward: float
    done: bool
    legal_actions: tuple[int, ...]
    info: dict


class BlackjackEnv:
    """One-suit blackjack environment from the agent player's perspective.

    Player index 0 is the learning agent. Other players are simulated only so
    their visible cards affect the shared physical deck and the observation.
    The simulator knows hidden cards because they are physically unavailable,
    but the observation list includes them only after they are revealed.
    """

    action_size = 3

    def __init__(
        self,
        num_players: int = 5,
        seed: int | None = None,
        opponent_policy=dealer_like_policy,
    ) -> None:
        if not 1 <= num_players <= 5:
            raise ValueError("num_players must be between 1 and 5")
        self.rng = random.Random(seed)
        self.num_players = num_players
        self.opponent_policy = opponent_policy
        self.deck = StandardDeck(self.rng)
        self.seen_cards_since_shuffle: list[int] = []
        self.round_cards: list[int] = []
        self.round_seen_cards: list[int] = []
        self.player_hands: list[list[int]] = []
        self.dealer_hand: list[int] = []
        self.dealer_hole_revealed = False
        self.bet = 1
        self.can_double = True
        self.done = True
        self.last_reward = 0.0

    @property
    def observation_size(self) -> int:
        return 25

    @property
    def agent_hand(self) -> list[int]:
        return self.player_hands[0]

    @property
    def dealer_upcard(self) -> int:
        return self.dealer_hand[0]

    def reset(self) -> tuple[float, ...]:
        """Deal a new round and return the first observable state.

        The existing deck is not reset here. This preserves the physical-game
        rule that shuffling happens only when the draw pile has been consumed.
        """

        self.round_cards = []
        self.round_seen_cards = []
        self.player_hands = [[] for _ in range(self.num_players)]
        self.dealer_hand = []
        self.dealer_hole_revealed = False
        self.bet = 1
        self.can_double = True
        self.done = False
        self.last_reward = 0.0

        for _ in range(2):
            for hand in self.player_hands:
                hand.append(self._draw(visible=True))
            self.dealer_hand.append(self._draw(visible=len(self.dealer_hand) == 0))

        return self.state()

    def legal_actions(self) -> tuple[int, ...]:
        """Return the actions currently allowed by the blackjack rules."""

        if self.done:
            return ()
        if self.can_double:
            return (HIT, STAND, DOUBLE)
        return (HIT, STAND)

    def step(self, action: int) -> StepResult:
        """Apply one player action and advance the round as needed.

        Hit keeps the turn alive unless the player busts. Stand resolves the
        dealer and comparison immediately. Double is allowed only while
        ``can_double`` is true, draws exactly one card, then resolves.
        """

        if self.done:
            raise RuntimeError("Cannot step a finished round; call reset()")
        if action not in self.legal_actions():
            raise ValueError(f"Illegal action {action}; legal actions are {self.legal_actions()}")

        if action == HIT:
            self.agent_hand.append(self._draw(visible=True))
            self.can_double = False
            if is_bust(self.agent_hand):
                return self._finish(-self.bet, {"result": "player_bust"})
            return StepResult(self.state(), 0.0, False, self.legal_actions(), {})

        if action == DOUBLE:
            self.bet = 2
            self.can_double = False
            self.agent_hand.append(self._draw(visible=True))
            if is_bust(self.agent_hand):
                return self._finish(-self.bet, {"result": "player_bust", "doubled": True})
            return self._resolve_round({"doubled": True})

        self.can_double = False
        return self._resolve_round({})

    def state(self) -> tuple[float, ...]:
        """Return the observable DQN state vector.

        The vector contains the player's hand summary, dealer upcard, whether
        double is still legal, estimated remaining deck composition, visible
        opponent-card composition, and table size. Values are normalized to
        keep neural-network inputs on a small numeric scale.
        """

        player_total = min(hand_value(self.agent_hand), 31) / 31.0
        soft = 1.0 if usable_ace(self.agent_hand) else 0.0
        dealer = self.dealer_upcard / 10.0
        can_double = 1.0 if self.can_double else 0.0

        known_remaining = known_remaining_counts(self.seen_cards_since_shuffle)
        remaining_features = tuple(
            count / fresh for count, fresh in zip(known_remaining, FRESH_VALUE_COUNTS)
        )

        opponent_cards = [
            card
            for index, hand in enumerate(self.player_hands)
            if index != 0
            for card in hand
        ]
        opponent_counts = value_counts(opponent_cards)
        opponent_features = tuple(
            min(count / max(fresh, 1), 1.0)
            for count, fresh in zip(opponent_counts, FRESH_VALUE_COUNTS)
        )

        opponents = (self.num_players - 1) / 4.0
        return (
            player_total,
            soft,
            dealer,
            can_double,
            *remaining_features,
            *opponent_features,
            opponents,
        )

    def _draw(self, visible: bool) -> int:
        """Draw a card and update the visible-card bookkeeping."""

        result = self.deck.draw(active_cards=self._active_cards())
        if result.reshuffled:
            self.seen_cards_since_shuffle = self._visible_active_cards()
        self.round_cards.append(result.value)
        if visible:
            self._remember_visible(result.value)
        return result.value

    def _resolve_round(self, info: dict) -> StepResult:
        """Finish all non-agent play, run the dealer, and score the hand."""

        self._play_opponents()
        # Reveal the dealer hole card only after all players have acted.
        self.dealer_hole_revealed = True
        self._remember_visible(self.dealer_hand[1])
        while hand_value(self.dealer_hand) < 17:
            self.dealer_hand.append(self._draw(visible=True))

        player_total = hand_value(self.agent_hand)
        dealer_total = hand_value(self.dealer_hand)
        if dealer_total > 21 or player_total > dealer_total:
            reward = self.bet
            result = "win"
        elif player_total < dealer_total:
            reward = -self.bet
            result = "loss"
        else:
            reward = 0
            result = "draw"
        return self._finish(reward, {**info, "result": result, "dealer_total": dealer_total})

    def _play_opponents(self) -> None:
        """Simulate other players before the dealer reveals the hidden card."""

        for hand in self.player_hands[1:]:
            while not is_bust(hand) and self.opponent_policy(hand) == HIT:
                hand.append(self._draw(visible=True))

    def _active_cards(self) -> list[int]:
        """Return all cards physically unavailable because they are on table."""

        return [card for hand in self.player_hands for card in hand] + list(self.dealer_hand)

    def _visible_active_cards(self) -> list[int]:
        """Return active table cards visible to the agent after a reshuffle."""

        visible = [card for hand in self.player_hands for card in hand]
        if self.dealer_hole_revealed:
            visible.extend(self.dealer_hand)
        elif self.dealer_hand:
            visible.append(self.dealer_hand[0])
        return visible

    def _remember_visible(self, card: int) -> None:
        """Record one newly visible physical card in the observable history."""

        self.seen_cards_since_shuffle.append(card)
        self.round_seen_cards.append(card)

    def _finish(self, reward: float, info: dict) -> StepResult:
        """Mark the round terminal and reveal any cards that were hidden.

        Hidden cards still came from the shared physical deck, but they are
        recorded in the agent observation only if normal play revealed them.
        A player bust can finish the round without leaking the dealer hole card.
        """

        self.done = True
        self.last_reward = float(reward)

        info = {
            **info,
            "player_total": hand_value(self.agent_hand),
            "dealer_hand": tuple(self.dealer_hand),
            "agent_hand": tuple(self.agent_hand),
            "bet": self.bet,
        }
        return StepResult(self.state(), float(reward), True, (), info)


def build_live_state(
    player_cards: list[int],
    dealer_upcard: int,
    visible_opponent_cards: list[int] | None = None,
    seen_cards_since_shuffle: list[int] | None = None,
    can_double: bool = True,
    num_players: int = 5,
) -> tuple[float, ...]:
    """Build the same state vector from manually entered physical-card data.

    This is used during live play, where the environment does not deal cards
    itself. The caller supplies all cards currently visible since the latest
    reshuffle, and the function mirrors ``BlackjackEnv.state``.
    """

    visible_opponent_cards = visible_opponent_cards or []
    seen_cards_since_shuffle = seen_cards_since_shuffle or []
    known_cards = list(seen_cards_since_shuffle) + list(player_cards) + [dealer_upcard]
    known_cards.extend(visible_opponent_cards)

    player_total = min(hand_value(player_cards), 31) / 31.0
    soft = 1.0 if usable_ace(player_cards) else 0.0
    dealer = dealer_upcard / 10.0
    can_double_feature = 1.0 if can_double else 0.0
    known_remaining = known_remaining_counts(known_cards)
    remaining_features = tuple(
        count / fresh for count, fresh in zip(known_remaining, FRESH_VALUE_COUNTS)
    )
    opponent_counts = value_counts(visible_opponent_cards)
    opponent_features = tuple(
        min(count / max(fresh, 1), 1.0)
        for count, fresh in zip(opponent_counts, FRESH_VALUE_COUNTS)
    )
    opponents = max(min(num_players - 1, 4), 0) / 4.0
    return (
        player_total,
        soft,
        dealer,
        can_double_feature,
        *remaining_features,
        *opponent_features,
        opponents,
    )
