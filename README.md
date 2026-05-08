# Blackjack-Agent
An implementation of a blackjack agent using deep reinforcement learning for the assignment in `TASK.md`.







The project models the assignment's blackjack variant:

- one-suit 13-card deck,
- up to five players,
- actions `hit`, `stand`, and `double`,
- `double` only as the first action,
- dealer draws until reaching at least 17,
- terminal rewards matching the assignment score: `+1`, `0`, `-1`, doubled to `+2`, `0`, `-2`.

## Reinforcement Learning Formulation

The agent is trained with DQN. The neural network receives an observable state vector and outputs three Q-values, one for each action.

State features:

- player hand total,
- usable-ace flag,
- dealer visible card,
- whether `double` is still legal,
- estimated remaining one-suit deck counts from visible cards,
- visible opponent-card counts,
- number of opponents.

Action space:

- `0 = hit`
- `1 = stand`
- `2 = double`

Illegal actions are masked, so `double` cannot be selected after the first action.

Rewards are sparse and terminal. Intermediate steps receive `0`; the final round result gives `+1`, `0`, or `-1`, with `double` changing this to `+2`, `0`, or `-2`.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Train

```bash
python -m scripts.train_dqn --episodes 50000
```

The default checkpoint is written to:

```text
models/dqn_blackjack.pt
```

## Evaluate

Evaluate the trained DQN:

```bash
python -m scripts.evaluate --episodes 10000
```

Evaluate the fixed baseline policy:

```bash
python -m scripts.evaluate --baseline --episodes 10000
```

## Live Play

After training, use the model during physical-card play:

```bash
python -m scripts.live_play
```

The CLI asks for your cards, the dealer upcard, visible opponent cards, cards previously seen since the last reshuffle, and whether this is your first action. It then prints the recommended action and the model's Q-values.

## License

This project is licensed under the PolyForm Noncommercial License 1.0.0.

You may use, study, and modify this project for personal, educational, and other non-commercial purposes.

Commercial use is not permitted. This includes selling the code, using it in paid products or services, or otherwise using it to generate revenue.
