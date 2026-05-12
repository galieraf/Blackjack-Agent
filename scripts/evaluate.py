"""Evaluate a trained DQN blackjack agent.

Evaluation runs complete rounds with exploration turned off. The optional
baseline mode uses a fixed policy instead of loading a neural network.
"""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

from agent.dqn import DQNAgent
from agent.tabular_dqn import TabularDQNAgent
from blackjack.actions import ACTION_NAMES
from blackjack.env import BlackjackEnv
from blackjack.policies import basic_strategy_policy, basic_training_policy


def parse_args() -> argparse.Namespace:
    """Read command-line evaluation settings."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--agent", choices=("dqn", "tabular"), default="dqn")
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--episodes", type=int, default=10_000)
    parser.add_argument("--num-players", type=int, default=5)
    parser.add_argument("--seed", type=int, default=19)
    parser.add_argument("--baseline", action="store_true", help="Evaluate the fixed baseline instead of DQN")
    parser.add_argument(
        "--baseline-policy",
        choices=("training", "strategy"),
        default="training",
        help="Fixed policy to use with --baseline",
    )
    return parser.parse_args()


def main() -> None:
    """Run evaluation episodes and print aggregate results."""

    args = parse_args()
    env = BlackjackEnv(num_players=args.num_players, seed=args.seed)
    if args.checkpoint is None:
        args.checkpoint = (
            Path("models/tabular_blackjack.json")
            if args.agent == "tabular"
            else Path("models/dqn_blackjack.pt")
        )
    if args.baseline:
        agent = None
    elif args.agent == "tabular":
        agent = TabularDQNAgent.load(args.checkpoint)
    else:
        agent = DQNAgent.load(str(args.checkpoint))

    total_reward = 0.0
    action_counts: Counter[str] = Counter()
    result_counts: Counter[str] = Counter()
    busts = 0

    for _ in range(args.episodes):
        state = env.reset()
        done = False
        while not done:
            if args.baseline:
                if args.baseline_policy == "strategy":
                    action = basic_strategy_policy(env.agent_hand, env.dealer_upcard, env.can_double)
                else:
                    action = basic_training_policy(env.agent_hand, env.can_double)
                # Keep the baseline inside the environment's legal action set.
                if action not in env.legal_actions():
                    action = env.legal_actions()[0]
            else:
                action = agent.select_action(state, env.legal_actions(), epsilon=0.0)
            action_counts[ACTION_NAMES[action]] += 1
            result = env.step(action)
            state = result.state
            done = result.done
        total_reward += result.reward
        result_counts[result.info.get("result", "unknown")] += 1
        if result.info.get("result") == "player_bust":
            busts += 1

    print(f"episodes={args.episodes}")
    agent_name = f"{args.baseline_policy}_baseline" if args.baseline else args.agent
    print(f"agent={agent_name}")
    print(f"average_reward={total_reward / args.episodes:.4f}")
    print(f"results={dict(result_counts)}")
    print(f"actions={dict(action_counts)}")
    print(f"win_rate={result_counts['win'] / args.episodes:.4f}")
    print(f"non_loss_rate={(result_counts['win'] + result_counts['draw']) / args.episodes:.4f}")
    print(f"bust_rate={busts / args.episodes:.4f}")


if __name__ == "__main__":
    main()
