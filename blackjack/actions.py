"""Action constants for the blackjack agent."""

HIT = 0
STAND = 1
DOUBLE = 2

ACTION_NAMES = {
    HIT: "hit",
    STAND: "stand",
    DOUBLE: "double",
}

NAME_TO_ACTION = {name: action for action, name in ACTION_NAMES.items()}
