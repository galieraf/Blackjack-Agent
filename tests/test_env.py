"""Regression tests for the assignment blackjack environment."""

from collections import Counter
import unittest

from blackjack.actions import DOUBLE, HIT, STAND
from blackjack.cards import FRESH_VALUE_COUNTS
from blackjack.env import BlackjackEnv, build_live_state


class EnvironmentTests(unittest.TestCase):
    """Check the most important rule constraints from TASK.md."""

    def test_reset_returns_valid_state(self):
        env = BlackjackEnv(seed=1)
        state = env.reset()
        self.assertEqual(len(state), env.observation_size)
        self.assertEqual(env.legal_actions(), (HIT, STAND, DOUBLE))

    def test_double_only_first_action(self):
        env = BlackjackEnv(seed=2)
        env.reset()
        result = env.step(HIT)
        if not result.done:
            self.assertNotIn(DOUBLE, result.legal_actions)

    def test_stand_finishes_round_with_terminal_reward(self):
        env = BlackjackEnv(seed=3)
        env.reset()
        result = env.step(STAND)
        self.assertTrue(result.done)
        self.assertIn(result.reward, {-1.0, 0.0, 1.0})

    def test_double_reward_is_bounded_by_doubled_bet(self):
        env = BlackjackEnv(seed=4)
        env.reset()
        result = env.step(DOUBLE)
        self.assertTrue(result.done)
        self.assertIn(result.reward, {-2.0, 0.0, 2.0})

    def test_random_actions_complete_many_rounds(self):
        env = BlackjackEnv(seed=5)
        for _ in range(200):
            env.reset()
            while not env.done:
                action = env.rng.choice(env.legal_actions())
                env.step(action)
            self.assertTrue(env.done)

    def test_randomized_turn_order_plays_earlier_opponents_before_agent(self):
        def one_hit_policy(cards):
            return HIT if len(cards) < 3 else STAND

        env = BlackjackEnv(seed=8, opponent_policy=one_hit_policy)
        found_agent_after_opponent = False
        for _ in range(50):
            env.reset()
            if env.agent_turn_position > 0:
                found_agent_after_opponent = True
                for player_index in env.turn_order[: env.agent_turn_position]:
                    self.assertGreaterEqual(len(env.player_hands[player_index]), 3)
                for player_index in env.turn_order[env.agent_turn_position + 1 :]:
                    self.assertEqual(len(env.player_hands[player_index]), 2)
                break

        self.assertTrue(found_agent_after_opponent)

    def test_fixed_turn_order_keeps_agent_first(self):
        env = BlackjackEnv(seed=9, randomize_player_order=False)
        env.reset()

        self.assertEqual(env.turn_order[0], 0)
        self.assertEqual(env.agent_turn_position, 0)

    def test_live_state_shape(self):
        state = build_live_state(
            player_cards=[1, 7],
            dealer_upcard=10,
            visible_opponent_cards=[2, 3, 10],
            seen_cards_since_shuffle=[4, 5],
            can_double=True,
            num_players=5,
        )
        self.assertEqual(len(state), BlackjackEnv().observation_size)

    def test_reshuffle_does_not_return_active_visible_cards_to_draw_pile(self):
        env = BlackjackEnv(num_players=1, seed=1)
        env.player_hands = [[9, 9, 9, 9]]
        env.dealer_hand = [2, 3]
        env.dealer_hole_revealed = False
        env.done = False
        env.deck.counts = [0] * len(FRESH_VALUE_COUNTS)

        drawn = env._draw(visible=True)

        self.assertNotEqual(drawn, 9)
        self.assertEqual(env.deck.counts[8], 0)

    def test_reshuffle_excludes_hidden_dealer_card_from_true_draw_pile(self):
        env = BlackjackEnv(num_players=1, seed=2)
        env.player_hands = [[1, 1, 1]]
        env.dealer_hand = [2, 1]
        env.dealer_hole_revealed = False
        env.done = False
        env.deck.counts = [0] * len(FRESH_VALUE_COUNTS)

        drawn = env._draw(visible=True)

        self.assertNotEqual(drawn, 1)
        self.assertEqual(env.deck.counts[0], 0)

    def test_hidden_dealer_card_is_not_in_agent_observation(self):
        env = BlackjackEnv(num_players=1)
        env.player_hands = [[2, 3]]
        env.dealer_hand = [4, 10]
        env.dealer_hole_revealed = False
        env.can_double = True
        env.done = False
        env.seen_cards_since_shuffle = [2, 3, 4]

        state = env.state()

        ten_value_remaining_feature = state[55]
        self.assertEqual(ten_value_remaining_feature, 1.0)
        self.assertNotIn(10, env.seen_cards_since_shuffle)

    def test_hidden_dealer_card_is_not_revealed_after_player_bust(self):
        env = BlackjackEnv(num_players=1)
        env.player_hands = [[10, 9, 5]]
        env.dealer_hand = [4, 10]
        env.dealer_hole_revealed = False
        env.can_double = False
        env.done = False
        env.seen_cards_since_shuffle = [10, 9, 5, 4]

        env._finish(-1, {"result": "player_bust"})

        self.assertEqual(Counter(env.seen_cards_since_shuffle)[10], 1)


if __name__ == "__main__":
    unittest.main()
