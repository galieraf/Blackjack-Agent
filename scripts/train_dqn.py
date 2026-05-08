"""Train a DQN blackjack agent."""

from __future__ import annotations

import argparse
from pathlib import Path
import random

from agent.dqn import DQNAgent, Transition
from blackjack.actions import ACTION_NAMES
from blackjack.env import BlackjackEnv


def linear_epsilon(step: int, start: float, end: float, decay_steps: int) -> float:
    if step >= decay_steps:
        return end
    fraction = step / max(decay_steps, 1)
    return start + fraction * (end - start)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes", type=int, default=50_000)
    parser.add_argument("--num-players", type=int, default=5)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--buffer-size", type=int, default=50_000)
    parser.add_argument("--target-update", type=int, default=500)
    parser.add_argument("--epsilon-start", type=float, default=1.0)
    parser.add_argument("--epsilon-end", type=float, default=0.05)
    parser.add_argument("--epsilon-decay-steps", type=int, default=30_000)
    parser.add_argument("--checkpoint", type=Path, default=Path("models/dqn_blackjack.pt"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    env = BlackjackEnv(num_players=args.num_players, seed=args.seed)
    agent = DQNAgent(
        state_size=env.observation_size,
        lr=args.lr,
        batch_size=args.batch_size,
        buffer_size=args.buffer_size,
        seed=args.seed,
    )

    args.checkpoint.parent.mkdir(parents=True, exist_ok=True)
    total_reward = 0.0
    losses: list[float] = []

    for episode in range(1, args.episodes + 1):
        state = env.reset()
        done = False
        episode_reward = 0.0
        while not done:
            epsilon = linear_epsilon(episode, args.epsilon_start, args.epsilon_end, args.epsilon_decay_steps)
            action = agent.select_action(state, env.legal_actions(), epsilon)
            result = env.step(action)
            agent.replay.push(
                Transition(
                    state=state,
                    action=action,
                    reward=result.reward,
                    next_state=result.state,
                    done=result.done,
                    legal_next_actions=result.legal_actions,
                )
            )
            loss = agent.train_step()
            if loss is not None:
                losses.append(loss)
            state = result.state
            done = result.done
            episode_reward += result.reward

        total_reward += episode_reward
        if episode % args.target_update == 0:
            agent.update_target()
        if episode % 1_000 == 0:
            avg_reward = total_reward / 1_000
            avg_loss = sum(losses[-1_000:]) / max(len(losses[-1_000:]), 1)
            print(
                f"episode={episode} avg_reward={avg_reward:.3f} "
                f"epsilon={epsilon:.3f} avg_loss={avg_loss:.4f}"
            )
            total_reward = 0.0

    agent.save(str(args.checkpoint))
    print(f"Saved checkpoint to {args.checkpoint}")
    print("Actions:", ", ".join(f"{idx}={name}" for idx, name in ACTION_NAMES.items()))


if __name__ == "__main__":
    main()
