"""Train a tabular Q-learning blackjack agent.

The agent uses the same legal-action masking and terminal rewards as the DQN
trainer, but stores action values in a dictionary keyed by discrete blackjack
features.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import random

from agent.tabular_dqn import TabularDQNAgent, TabularTransition
from blackjack.actions import ACTION_NAMES
from blackjack.env import BlackjackEnv


def linear_epsilon(step: int, start: float, end: float, decay_steps: int) -> float:
    """Linearly reduce exploration from ``start`` to ``end``."""

    if step >= decay_steps:
        return end
    fraction = step / max(decay_steps, 1)
    return start + fraction * (end - start)


def parse_args() -> argparse.Namespace:
    """Read command-line training settings."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes", type=int, default=100_000)
    parser.add_argument("--num-players", type=int, default=5)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--alpha", type=float, default=0.08)
    parser.add_argument("--epsilon-start", type=float, default=1.0)
    parser.add_argument("--epsilon-end", type=float, default=0.02)
    parser.add_argument("--epsilon-decay-steps", type=int, default=70_000)
    parser.add_argument("--checkpoint", type=Path, default=Path("models/tabular_blackjack.json"))
    return parser.parse_args()


def main() -> None:
    """Train the tabular agent and write a checkpoint."""

    args = parse_args()
    random.seed(args.seed)
    env = BlackjackEnv(num_players=args.num_players, seed=args.seed)
    agent = TabularDQNAgent(alpha=args.alpha, seed=args.seed)

    args.checkpoint.parent.mkdir(parents=True, exist_ok=True)
    total_reward = 0.0
    td_errors: list[float] = []

    for episode in range(1, args.episodes + 1):
        state = env.reset()
        done = False
        episode_reward = 0.0
        epsilon = linear_epsilon(episode, args.epsilon_start, args.epsilon_end, args.epsilon_decay_steps)
        while not done:
            action = agent.select_action(state, env.legal_actions(), epsilon)
            result = env.step(action)
            td_error = agent.update(
                TabularTransition(
                    state=state,
                    action=action,
                    reward=result.reward,
                    next_state=result.state,
                    done=result.done,
                    legal_next_actions=result.legal_actions,
                )
            )
            td_errors.append(abs(td_error))
            state = result.state
            done = result.done
            episode_reward += result.reward

        total_reward += episode_reward
        if episode % 1_000 == 0:
            avg_reward = total_reward / 1_000
            avg_td_error = sum(td_errors[-1_000:]) / max(len(td_errors[-1_000:]), 1)
            print(
                f"episode={episode} avg_reward={avg_reward:.3f} "
                f"epsilon={epsilon:.3f} avg_abs_td_error={avg_td_error:.4f} "
                f"states={len(agent.q_table)}"
            )
            total_reward = 0.0

    agent.save(args.checkpoint)
    print(f"Saved checkpoint to {args.checkpoint}")
    print("Training target: tabular Q-learning")
    print("Actions:", ", ".join(f"{idx}={name}" for idx, name in ACTION_NAMES.items()))


if __name__ == "__main__":
    main()
