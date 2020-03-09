# 2048 Solver

This project is a python powered 2048 solver. The move logic is retrieved through a breadth-first-search [Expectiminimax](https://en.wikipedia.org/wiki/Expectiminimax) tree with [alpha-beta-pruning](https://en.wikipedia.org/wiki/Alpha%E2%80%93beta_pruning). (See below for summary on these) Using selenium to connect controls, the game is then read by PlayerAI_3 to decide each move.

Original Game: https://play2048.co/

## Installation

- using virtualenv is recommended
- internet connection will be necessary for the chrome launcher (chrome may be cached after the first start up)

> pip install -r requirements.txt

## Expectiminimax

If you are playing chess, you want your next move to give you the best odds at a winning position. At the same time you must consider that your
opponent will move in such a way to decrease your odds as much as possible. This is a normal mini-max situation. Since each time you move in 2048, a new 2 or 4 tile will appear in a random location, you decide your move considering some chance (not against a player who may have a clear best move). Expectiminimax is an altered version of normal zero-sum decision trees to take this into account.

## alpha-beta-pruning

The idea is that once you have your expanded game state tree, it may be too large to traverse every node. Think about how many neighbor states a board has considering you can move up, down, left, or right and for each of those moves you may have up to 14 random new tile states. To help traverse the tree efficiently, you can "prune" branches of the tree with weak scores.
