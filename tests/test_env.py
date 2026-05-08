"""Regression tests for the assignment blackjack environment."""

import unittest

from blackjack.actions import DOUBLE, HIT, STAND
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

    def test_live_state_shape(self):
        state = build_live_state(
            player_cards=[1, 7],
            dealer_upcard=10,
            visible_opponent_cards=[2, 3, 10],
            seen_cards_since_shuffle=[4, 5],
            can_double=True,
            num_players=5,
        )
        self.assertEqual(len(state), 25)


if __name__ == "__main__":
    unittest.main()
