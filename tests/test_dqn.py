"""Regression tests for DQN and Double DQN target computation."""

import unittest

from agent import dqn
from agent.dqn import DQNAgent, Transition
from agent.tabular_dqn import TabularDQNAgent, TabularTransition
from blackjack.env import BlackjackEnv


def sample_state(total: int = 12, dealer: int = 10, can_double: bool = True) -> tuple[float, ...]:
    """Build a minimal valid 25-feature environment state."""

    env = BlackjackEnv(num_players=5, randomize_player_order=False)
    second_card = total - 2 if total <= 11 else total - 10
    env.player_hands = [[2 if total <= 11 else 10, second_card]]
    env.dealer_hand = [dealer, 10]
    env.dealer_hole_revealed = False
    env.can_double = can_double
    env.done = False
    env.seen_cards_since_shuffle = list(env.agent_hand) + [dealer]
    env.turn_order = list(range(env.num_players))
    env.agent_turn_position = 0
    return env.state()


class TabularDQNTests(unittest.TestCase):
    """Check the tabular comparison agent's masking and update behavior."""

    def test_tabular_action_selection_masks_illegal_actions(self):
        agent = TabularDQNAgent(seed=1)
        state = sample_state()
        agent.q_table[agent.state_key(state)] = [0.0, 1.0, 5.0]

        action = agent.select_action(state, legal_actions=(0, 1), epsilon=0.0)

        self.assertEqual(action, 1)

    def test_tabular_update_uses_best_legal_next_action(self):
        agent = TabularDQNAgent(alpha=1.0)
        state = sample_state(total=10)
        next_state = sample_state(total=15, can_double=False)
        agent.q_table[agent.state_key(next_state)] = [2.0, 3.0, 99.0]

        agent.update(
            TabularTransition(
                state=state,
                action=0,
                reward=1.0,
                next_state=next_state,
                done=False,
                legal_next_actions=(0, 1),
            )
        )

        self.assertEqual(agent.q_values(state)[0], 4.0)


@unittest.skipIf(dqn.torch is None, "PyTorch is not installed")
class DQNTargetTests(unittest.TestCase):
    """Check that future-value targets preserve illegal action masking."""

    def _transition(self, legal_next_actions: tuple[int, ...], done: bool = False) -> Transition:
        return Transition(
            state=(0.0,),
            action=0,
            reward=0.0,
            next_state=(0.0,),
            done=done,
            legal_next_actions=legal_next_actions,
        )

    def test_regular_dqn_uses_best_legal_target_network_value(self):
        agent = DQNAgent(state_size=1, batch_size=1, double_dqn=False)
        next_states = dqn.torch.tensor([[0.0]], dtype=dqn.torch.float32, device=agent.device)
        dones = dqn.torch.tensor([False], dtype=dqn.torch.bool, device=agent.device)

        class TargetNet(dqn.nn.Module):
            def forward(self, x):
                return dqn.torch.tensor([[1.0, 5.0, 9.0]], device=x.device)

        agent.target = TargetNet().to(agent.device)
        value = agent._next_state_values(next_states, dones, [self._transition((0, 1))])

        self.assertEqual(float(value.item()), 5.0)

    def test_double_dqn_selects_with_online_and_evaluates_with_target(self):
        agent = DQNAgent(state_size=1, batch_size=1, double_dqn=True)
        next_states = dqn.torch.tensor([[0.0]], dtype=dqn.torch.float32, device=agent.device)
        dones = dqn.torch.tensor([False], dtype=dqn.torch.bool, device=agent.device)

        class OnlineNet(dqn.nn.Module):
            def forward(self, x):
                return dqn.torch.tensor([[1.0, 9.0, 2.0]], device=x.device)

        class TargetNet(dqn.nn.Module):
            def forward(self, x):
                return dqn.torch.tensor([[10.0, 3.0, 8.0]], device=x.device)

        agent.online = OnlineNet().to(agent.device)
        agent.target = TargetNet().to(agent.device)
        value = agent._next_state_values(next_states, dones, [self._transition((0, 1, 2))])

        self.assertEqual(float(value.item()), 3.0)

    def test_double_dqn_masks_illegal_online_selection(self):
        agent = DQNAgent(state_size=1, batch_size=1, double_dqn=True)
        next_states = dqn.torch.tensor([[0.0]], dtype=dqn.torch.float32, device=agent.device)
        dones = dqn.torch.tensor([False], dtype=dqn.torch.bool, device=agent.device)

        class OnlineNet(dqn.nn.Module):
            def forward(self, x):
                return dqn.torch.tensor([[1.0, 4.0, 99.0]], device=x.device)

        class TargetNet(dqn.nn.Module):
            def forward(self, x):
                return dqn.torch.tensor([[10.0, 3.0, 100.0]], device=x.device)

        agent.online = OnlineNet().to(agent.device)
        agent.target = TargetNet().to(agent.device)
        value = agent._next_state_values(next_states, dones, [self._transition((0, 1))])

        self.assertEqual(float(value.item()), 3.0)

    def test_terminal_next_state_has_zero_future_value(self):
        agent = DQNAgent(state_size=1, batch_size=1, double_dqn=True)
        next_states = dqn.torch.tensor([[0.0]], dtype=dqn.torch.float32, device=agent.device)
        dones = dqn.torch.tensor([True], dtype=dqn.torch.bool, device=agent.device)

        value = agent._next_state_values(next_states, dones, [self._transition((), done=True)])

        self.assertEqual(float(value.item()), 0.0)


if __name__ == "__main__":
    unittest.main()
