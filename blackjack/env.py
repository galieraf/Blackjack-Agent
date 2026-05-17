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
    """Standard-pack blackjack environment from the agent player's perspective.

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
        randomize_player_order: bool = True,
    ) -> None:
        if not 1 <= num_players <= 5:
            raise ValueError("num_players must be between 1 and 5")
        self.rng = random.Random(seed)
        self.num_players = num_players
        self.opponent_policy = opponent_policy
        self.randomize_player_order = randomize_player_order
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
        self.turn_order: list[int] = []
        self.agent_turn_position = 0

    @property
    def observation_size(self) -> int:
        return 71

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

        self.turn_order = list(range(self.num_players))
        if self.randomize_player_order:
            self.rng.shuffle(self.turn_order)
        self.agent_turn_position = self.turn_order.index(0)
        self._play_opponents_before_agent()

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

        opponent_cards = [
            card
            for index, hand in enumerate(self.player_hands)
            if index != 0
            for card in hand
        ]
        return _build_observation(
            player_cards=self.agent_hand,
            dealer_upcard=self.dealer_upcard,
            visible_opponent_cards=opponent_cards,
            known_visible_cards=self.seen_cards_since_shuffle,
            can_double=self.can_double,
            num_players=self.num_players,
            agent_turn_position=self.agent_turn_position,
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

        self._play_opponents_after_agent()
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

    def _play_opponents_before_agent(self) -> None:
        """Simulate opponents whose randomized turns come before the agent."""

        for player_index in self.turn_order[: self.agent_turn_position]:
            self._play_opponent(player_index)

    def _play_opponents_after_agent(self) -> None:
        """Simulate opponents whose randomized turns come after the agent."""

        if not self.turn_order:
            self.turn_order = list(range(self.num_players))
            self.agent_turn_position = self.turn_order.index(0)
        for player_index in self.turn_order[self.agent_turn_position + 1 :]:
            self._play_opponent(player_index)

    def _play_opponent(self, player_index: int) -> None:
        """Play one non-agent hand using the configured opponent policy."""

        if player_index == 0:
            return
        hand = self.player_hands[player_index]
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
    agent_turn_position: int = 0,
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
    return _build_observation(
        player_cards=player_cards,
        dealer_upcard=dealer_upcard,
        visible_opponent_cards=visible_opponent_cards,
        known_visible_cards=known_cards,
        can_double=can_double,
        num_players=num_players,
        agent_turn_position=agent_turn_position,
    )


def _one_hot(value: int, size: int) -> tuple[float, ...]:
    """Return a simple one-hot vector for already clipped integer values."""

    return tuple(1.0 if index == value else 0.0 for index in range(size))


def _build_observation(
    player_cards: list[int],
    dealer_upcard: int,
    visible_opponent_cards: list[int],
    known_visible_cards: list[int],
    can_double: bool,
    num_players: int,
    agent_turn_position: int,
) -> tuple[float, ...]:
    """
    Build the expanded DQN observation used by training and live play.

    State is a fixed-size numeric vector with 71 features.
    It represents the currently observable blackjack situation:
    - my hand
    - dealer's visible card
    - whether double is allowed
    - estimated remaining deck composition
    - visible opponent cards
    - number of players
    - my turn position
    """

    # ===== 1) My hand features =====

    # Current value of my hand.
    # hand_value() correctly handles aces as 1 or 11.
    # min(..., 31) limits very large busted values.
    total = min(hand_value(player_cards), 31)

    # Normalized hand value.
    # Example: total = 14 -> 14 / 31.
    # State index: 0
    player_total = total / 31.0

    # Whether I have a usable ace.
    # usable ace = ace that can currently be counted as 11 without busting.
    # Example: A + 7 = soft 18 -> soft = 1.0
    # State index: 1
    soft = 1.0 if usable_ace(player_cards) else 0.0

    # Whether double is currently legal.
    # Usually double is allowed only as the first action.
    # State index: 2
    can_double_feature = 1.0 if can_double else 0.0

    # Number of cards in my hand, normalized.
    # Capped at 8 cards.
    # State index: 3
    player_cards_feature = min(len(player_cards), 8) / 8.0

    # One-hot encoding of my hand total.
    # Length = 32, for totals 0..31.
    # Example: total = 14 -> position 14 is 1, others are 0.
    # State indices: 4..35
    total_one_hot = _one_hot(total, 32)


    # ===== 2) Dealer visible card =====

    # One-hot encoding of dealer's upcard.
    # Length = 10:
    # index 0 = Ace
    # index 1 = 2
    # index 2 = 3
    # ...
    # index 8 = 9
    # index 9 = 10 / J / Q / K
    #
    # max(1, min(10, dealer_upcard)) makes sure the card value is in range 1..10.
    # State indices: 36..45
    dealer_one_hot = _one_hot(max(1, min(10, dealer_upcard)) - 1, 10)


    # ===== 3) Estimated remaining deck features =====

    # Estimate how many cards of each value are still unseen.
    # This is based only on known visible cards, not hidden cards.
    #
    # Values are grouped as:
    # A, 2, 3, 4, 5, 6, 7, 8, 9, 10-value cards
    #
    # 10-value cards include 10, J, Q, K.
    known_remaining = known_remaining_counts(known_visible_cards)

    # Remaining card ratios for each value group.
    # Each count is divided by the number of such cards in a fresh deck.
    #
    # Example:
    # If there are 4 aces in a fresh deck and 3 are estimated to remain,
    # feature = 3 / 4 = 0.75.
    #
    # State indices: 46..55
    remaining_features = tuple(
        count / fresh
        for count, fresh in zip(known_remaining, FRESH_VALUE_COUNTS)
    )

    # Fraction of total cards still unseen.
    # Example: if half of the deck is estimated to remain, this is about 0.5.
    # State index: 56
    remaining_cards = sum(known_remaining) / sum(FRESH_VALUE_COUNTS)

    # Count visible cards by value group:
    # index 0 = Ace
    # index 1 = 2
    # ...
    # index 9 = 10 / J / Q / K
    seen_counts = value_counts(known_visible_cards)

    # Simplified Hi-Lo card counting feature.
    #
    # Low cards 2..6 increase the count.
    # High cards A and 10-value cards decrease the count.
    #
    # If hi_lo is positive:
    #   many low cards have already been seen,
    #   so the remaining deck is relatively richer in high cards.
    #
    # If hi_lo is negative:
    #   many high cards have already been seen,
    #   so the remaining deck is relatively poorer in high cards.
    hi_lo = sum(seen_counts[1:6]) - seen_counts[0] - seen_counts[9]

    # Normalized true-count-like feature.
    #
    # The raw Hi-Lo count is divided by the estimated remaining deck fraction,
    # then clipped to [-10, 10], then divided by 10.
    #
    # Final value is approximately in range [-1, 1].
    # State index: 57
    true_count = max(
        -10.0,
        min(
            10.0,
            hi_lo / max(remaining_cards, 1 / sum(FRESH_VALUE_COUNTS))
        )
    ) / 10.0


    # ===== 4) Visible opponent cards =====

    # Count visible cards of other players by value group.
    # Again grouped as:
    # A, 2, 3, 4, 5, 6, 7, 8, 9, 10-value cards.
    opponent_counts = value_counts(visible_opponent_cards)

    # Normalized visible opponent card counts.
    #
    # These features tell the agent what cards other players have shown.
    # This is useful because those cards are no longer available in the deck.
    #
    # State indices: 58..67
    opponent_features = tuple(
        min(count / max(fresh, 1), 1.0)
        for count, fresh in zip(opponent_counts, FRESH_VALUE_COUNTS)
    )

    # Total number of visible opponent cards, normalized.
    # Capped at 20 cards.
    # State index: 68
    opponent_cards_feature = min(len(visible_opponent_cards), 20) / 20.0


    # ===== 5) Table/player-position features =====

    # Number of opponents, normalized.
    #
    # num_players includes the agent.
    # So if there are 5 players total:
    # opponents = (5 - 1) / 4 = 1.0
    #
    # If there are 3 players total:
    # opponents = (3 - 1) / 4 = 0.5
    #
    # State index: 69
    opponents = max(min(num_players - 1, 4), 0) / 4.0

    # Agent's turn position at the table, normalized.
    #
    # 0.00 = agent acts first
    # 0.25 = second
    # 0.50 = third
    # 0.75 = fourth
    # 1.00 = fifth
    #
    # State index: 70
    turn_position = max(min(agent_turn_position, 4), 0) / 4.0


    # ===== Final 71-dimensional state vector =====

    return (
        # 0..3: compact features of my hand and legal double
        player_total,
        soft,
        can_double_feature,
        player_cards_feature,

        # 4..35: one-hot encoding of my hand total
        *total_one_hot,

        # 36..45: one-hot encoding of dealer's upcard
        *dealer_one_hot,

        # 46..55: estimated remaining deck composition
        *remaining_features,

        # 56: fraction of cards remaining
        remaining_cards,

        # 57: simplified Hi-Lo / true-count feature
        true_count,

        # 58..67: visible opponent cards by value
        *opponent_features,

        # 68: total number of visible opponent cards
        opponent_cards_feature,

        # 69: number of opponents / table size
        opponents,

        # 70: my turn position
        turn_position,
    )
