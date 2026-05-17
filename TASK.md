# Blackjack Agent Assignment
Your task is to write an agent for playing blackjack. In the last exercise session of the semester, we will play blackjack with your agents using physical cards. The computer will tell you which action to take. We will use the actions **hit**, **stand**, and **double**. This means that your agent needs some way to receive information about the current state of the table, ideally including not only your own cards and the dealer’s visible card, but also your opponents’ cards.

Compared to casino blackjack, we will play with one standard 52-card deck. Card suits do not affect play, so the simulator represents the deck only by blackjack values: four Aces, four of each 2-9, and sixteen 10-value cards (10, Jack, Queen, King). Each table will have at most five players, that is, five students. The deck will be reshuffled only after we run out of cards.

In each round, the initial bet is one unit. You may double the bet by choosing the action **double**.

We recommend using reinforcement learning to build your agent, as this should help you better understand the material for the subsequent exam. However, you may also use other methods, provided that you can explain them in detail. This requirement also applies to reinforcement-learning-based methods: you must be able to explain your approach clearly and in detail.

## Scoring
We will play 20 rounds. If your score is within 10 points of the best player at your table, you will receive 25 points. If your score is within 11–15 points of the best player at your table, you will receive 20 points. If your score is 16 or more points below the best player at your table, you will receive 15 points.

You will also be asked three questions about your method. Each incorrect explanation will result in a deduction of 5 points. A good strategy is therefore to choose a method that you understand well and that still performs reasonably well. For example, standard tabular reinforcement learning methods covered in class could work quite well, but you are also free to use deep reinforcement learning.

## Rules of the Blackjack Variant
The goal of blackjack is to obtain a hand value as close as possible to 21 without exceeding it. A hand with value greater than 21 is called a **bust** and loses automatically.

Cards have the following values:
- Number cards have their face value.
- Jack, Queen, and King each have value 10.
- An Ace can have value 1 or 11, whichever is better for the hand without causing it to bust. 

At the beginning of each round, every player places a bet of one unit. Each player is dealt two cards. The dealer is also dealt two cards: one dealer card is visible to all players, and the other dealer card remains hidden until all players have finished their turns.


Players then act one by one. On your turn, your agent must choose one of the following three actions:

## Available Actions
### Hit

Take one additional card. If the value of your hand exceeds 21, you bust and lose the round immediately.

### Stand

Take no more cards. Your current hand value is kept, and your turn ends.

### Double

Double your bet from one unit to two units, take exactly one additional card, and then stand automatically. This action can be used only as your first action in the round.

The actions **split**, **surrender**, and **insurance** are not used in this variant.

## Dealer's Turn
After all players have finished their turns, the dealer reveals the hidden card and plays according to the standard dealer rule: the dealer draws cards until reaching a value of at least 17, and then stands.

## Round Outcome
The outcome of each player’s hand is determined by comparing it with the dealer’s hand:

- If the player busts, the player loses the bet.
- If the dealer busts and the player has not busted, the player wins the bet.
- If neither busts, the hand with the higher value wins.
- If the player and dealer have the same value, the result is a draw and no points are won or lost.

A normal win gives the player **+1 point**, and a normal loss gives **-1 point**. If the player used **double**, the result is doubled: a win gives **+2 points**, and a loss gives **-2 points**. A draw gives **0 points**.
