"""Regression tests for card parsing and ace scoring."""

import unittest

from blackjack.cards import hand_value, is_bust, parse_card, parse_cards, usable_ace


class CardTests(unittest.TestCase):
    """Verify the value model used by the one-suit deck."""

    def test_parse_cards(self):
        self.assertEqual(parse_cards("A 10 J Q K"), [1, 10, 10, 10, 10])
        self.assertEqual(parse_card("7"), 7)

    def test_soft_ace_scoring(self):
        self.assertEqual(hand_value([1, 7]), 18)
        self.assertTrue(usable_ace([1, 7]))

    def test_ace_downgrades_to_avoid_bust(self):
        self.assertEqual(hand_value([1, 9, 9]), 19)
        self.assertFalse(usable_ace([1, 9, 9]))
        self.assertFalse(is_bust([1, 9, 9]))

    def test_bust_detection(self):
        self.assertTrue(is_bust([10, 9, 3]))


if __name__ == "__main__":
    unittest.main()
