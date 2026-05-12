"""Train a DQN blackjack agent.

Each episode is one blackjack round. The script stores transitions with the
next state's legal actions so the DQN target respects the double rule.
Double DQN is the default because it usually gives less over-optimistic action
values in this small sparse-reward game.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import random

from agent.dqn import DQNAgent, Transition
from blackjack.actions import ACTION_NAMES
from blackjack.env import BlackjackEnv
from blackjack.policies import basic_strategy_policy, basic_training_policy


def linear_epsilon(step: int, start: float, end: float, decay_steps: int) -> float:
    """Linearly reduce exploration from ``start`` to ``end``."""

    if step >= decay_steps:
        return end
    fraction = step / max(decay_steps, 1)
    return start + fraction * (end - start)


def parse_args() -> argparse.Namespace:
    """Read command-line training settings."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes", type=int, default=0)
    parser.add_argument("--num-players", type=int, default=5)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--buffer-size", type=int, default=50_000)
    parser.add_argument("--target-update", type=int, default=500)
    parser.add_argument("--train-steps-per-env-step", type=int, default=1)
    parser.add_argument("--imitation-episodes", type=int, default=20_000)
    parser.add_argument("--imitation-updates", type=int, default=3_000)
    parser.add_argument("--teacher-regularization-updates", type=int, default=1)
    parser.add_argument("--prefill-episodes", type=int, default=0)
    parser.add_argument("--prefill-policy", choices=("strategy", "training"), default="strategy")
    parser.add_argument("--epsilon-start", type=float, default=1.0)
    parser.add_argument("--epsilon-end", type=float, default=0.02)
    parser.add_argument("--epsilon-decay-steps", type=int, default=50_000)
    parser.add_argument("--double-dqn", dest="double_dqn", action="store_true", default=True)
    parser.add_argument("--regular-dqn", dest="double_dqn", action="store_false")
    parser.add_argument("--checkpoint", type=Path, default=Path("models/dqn_blackjack.pt"))
    return parser.parse_args()


def baseline_action(env: BlackjackEnv, policy_name: str) -> int:
    """Return the fixed policy action clipped to the current legal set."""

    if policy_name == "strategy":
        action = basic_strategy_policy(env.agent_hand, env.dealer_upcard, env.can_double)
    else:
        action = basic_training_policy(env.agent_hand, env.can_double)
    legal_actions = env.legal_actions()
    if action not in legal_actions:
        action = legal_actions[0]
    return action


def store_transition(
    agent: DQNAgent,
    state: tuple[float, ...],
    action: int,
    result,
) -> None:
    """Add one environment transition to replay memory."""

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


def run_training_updates(agent: DQNAgent, steps: int) -> list[float]:
    """Run up to ``steps`` gradient updates and return available losses."""

    losses: list[float] = []
    for _ in range(max(steps, 0)):
        loss = agent.train_step()
        if loss is not None:
            losses.append(loss)
    return losses


def prefill_from_baseline(
    env: BlackjackEnv,
    agent: DQNAgent,
    episodes: int,
    policy_name: str,
    train_steps_per_env_step: int,
    target_update: int,
) -> list[float]:
    """Seed replay with fixed-policy experience before epsilon exploration."""

    losses: list[float] = []
    transitions = 0
    for _ in range(episodes):
        state = env.reset()
        done = False
        while not done:
            action = baseline_action(env, policy_name)
            result = env.step(action)
            store_transition(agent, state, action, result)
            losses.extend(run_training_updates(agent, train_steps_per_env_step))
            transitions += 1
            if transitions % target_update == 0:
                agent.update_target()
            state = result.state
            done = result.done
    agent.update_target()
    return losses


def collect_teacher_samples(
    env: BlackjackEnv,
    episodes: int,
    policy_name: str,
) -> list[tuple[tuple[float, ...], int]]:
    """Generate state/action examples from a fixed policy."""

    samples: list[tuple[tuple[float, ...], int]] = []
    for _ in range(episodes):
        state = env.reset()
        done = False
        while not done:
            action = baseline_action(env, policy_name)
            samples.append((state, action))
            result = env.step(action)
            state = result.state
            done = result.done
    return samples


def run_imitation_warmup(
    agent: DQNAgent,
    samples: list[tuple[tuple[float, ...], int]],
    updates: int,
) -> list[float]:
    """Warm-start DQN action preferences from teacher examples."""

    if not samples or updates <= 0:
        return []
    losses: list[float] = []
    for _ in range(updates):
        batch = agent.rng.sample(samples, min(agent.batch_size, len(samples)))
        losses.append(agent.supervised_step(batch))
    agent.update_target()
    return losses


def main() -> None:
    """Train the agent and write a checkpoint."""

    args = parse_args()
    random.seed(args.seed)
    env = BlackjackEnv(num_players=args.num_players, seed=args.seed)
    agent = DQNAgent(
        state_size=env.observation_size,
        lr=args.lr,
        batch_size=args.batch_size,
        buffer_size=args.buffer_size,
        seed=args.seed,
        double_dqn=args.double_dqn,
    )

    args.checkpoint.parent.mkdir(parents=True, exist_ok=True)
    total_reward = 0.0
    teacher_samples = collect_teacher_samples(env, args.imitation_episodes, args.prefill_policy)
    imitation_losses = run_imitation_warmup(agent, teacher_samples, args.imitation_updates)
    if imitation_losses:
        avg_imitation_loss = sum(imitation_losses[-100:]) / min(len(imitation_losses), 100)
        print(
            f"imitation_episodes={args.imitation_episodes} "
            f"samples={len(teacher_samples)} avg_loss={avg_imitation_loss:.4f}"
        )
    losses = prefill_from_baseline(
        env=env,
        agent=agent,
        episodes=args.prefill_episodes,
        policy_name=args.prefill_policy,
        train_steps_per_env_step=args.train_steps_per_env_step,
        target_update=args.target_update,
    )
    if args.prefill_episodes:
        print(
            f"prefill_episodes={args.prefill_episodes} "
            f"replay_size={len(agent.replay)} losses={len(losses)}"
        )

    for episode in range(1, args.episodes + 1):
        state = env.reset()
        done = False
        episode_reward = 0.0
        epsilon = linear_epsilon(episode, args.epsilon_start, args.epsilon_end, args.epsilon_decay_steps)
        while not done:
            action = agent.select_action(state, env.legal_actions(), epsilon)
            result = env.step(action)
            # The environment supplies legal actions for result.state; replay
            # keeps them so train_step can mask illegal bootstrap actions.
            store_transition(agent, state, action, result)
            losses.extend(run_training_updates(agent, args.train_steps_per_env_step))
            if teacher_samples:
                for _ in range(args.teacher_regularization_updates):
                    batch = agent.rng.sample(teacher_samples, min(agent.batch_size, len(teacher_samples)))
                    agent.supervised_step(batch)
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
    print(f"Training target: {'Double DQN' if args.double_dqn else 'DQN'}")
    print("Actions:", ", ".join(f"{idx}={name}" for idx, name in ACTION_NAMES.items()))


if __name__ == "__main__":
    main()
