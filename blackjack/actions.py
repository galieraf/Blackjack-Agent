"""Action constants for the blackjack agent.

The integer ids are the action indexes produced by the neural network, so the
order must stay consistent between the environment, training, and live play.
"""

HIT = 0
STAND = 1
DOUBLE = 2

ACTION_NAMES = {
    HIT: "hit",
    STAND: "stand",
    DOUBLE: "double",
}

NAME_TO_ACTION = {name: action for action, name in ACTION_NAMES.items()}
