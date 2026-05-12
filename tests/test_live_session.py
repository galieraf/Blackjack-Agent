"""Tests for stateful physical-card live session tracking."""

import unittest

from blackjack.cards import known_remaining_counts
from scripts.live_session import LiveSessionState


class LiveSessionStateTests(unittest.TestCase):
    def test_round_end_moves_current_cards_to_seen_once(self):
        session = LiveSessionState(num_players=2)
        session.deal([1, 7], 10, [5, 6])
        session.me_hit(2)
        session.dealer_reveal(9)

        session.round_end()

        self.assertEqual(session.seen_cards_since_shuffle, [1, 7, 2, 10, 9, 5, 6])
        self.assertEqual(session.current_visible_cards(), [])
        self.assertEqual(session.known_cards_for_observation(), session.seen_cards_since_shuffle)

    def test_new_round_keeps_seen_without_duplicating_current_cards(self):
        session = LiveSessionState(num_players=2)
        session.deal([1, 7], 10, [5])
        session.round_end()
        session.deal([2, 3], 4, [6])

        self.assertEqual(session.seen_cards_since_shuffle, [1, 7, 10, 5])
        self.assertEqual(session.known_cards_for_observation(), [1, 7, 10, 5, 2, 3, 4, 6])

    def test_shuffle_clears_history_and_current_table(self):
        session = LiveSessionState(num_players=2)
        session.deal([1, 7], 10, [5])
        session.round_end()
        session.deal([2, 3], 4, [6])

        session.shuffle()

        self.assertEqual(session.seen_cards_since_shuffle, [])
        self.assertEqual(session.current_visible_cards(), [])

    def test_actions_update_double_legality(self):
        session = LiveSessionState(num_players=1)
        session.deal([5, 6], 10, [])
        self.assertEqual(session.legal_actions(), (0, 1, 2))

        session.me_hit(2)

        self.assertEqual(session.legal_actions(), (0, 1))

    def test_stand_ends_turn_and_removes_all_legal_actions(self):
        session = LiveSessionState(num_players=1)
        session.deal([10, 7], 6, [])

        session.me_stand()

        self.assertEqual(session.legal_actions(), ())
        with self.assertRaises(ValueError):
            session.model_state()

    def test_double_ends_turn_and_prevents_later_hit(self):
        session = LiveSessionState(num_players=1)
        session.deal([5, 6], 10, [])

        session.me_double(10)

        self.assertEqual(session.legal_actions(), ())
        with self.assertRaises(ValueError):
            session.me_hit(2)

    def test_remaining_counts_use_history_plus_current_visible_cards(self):
        session = LiveSessionState(num_players=2)
        session.seen_cards_since_shuffle = [1, 10]
        session.deal([1, 7], 10, [5])

        remaining = known_remaining_counts(session.known_cards_for_observation())

        self.assertEqual(remaining[0], 2)
        self.assertEqual(remaining[4], 3)
        self.assertEqual(remaining[6], 3)
        self.assertEqual(remaining[9], 14)


if __name__ == "__main__":
    unittest.main()
