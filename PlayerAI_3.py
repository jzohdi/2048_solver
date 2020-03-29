# -*- coding: utf-8 -*-
"""
Created on Fri Jul  6 22:25:11 2018

@author: jake
"""

import time
import math
from Grid_3 import Grid
from random import randint, uniform
from BaseAI_3 import BaseAI
from collections import deque
from typing import List, Set, Dict, Tuple

# from Dequeue import deque
probability = 0.8
possibleNewTiles = [2, 4]
time_limit = 2
l_rate = 0.5
# heuristic functions
#


def ease_slope(number):
    if number < 0:
        return -((-number)**(1/2))
    return number**(1/2)


class Heuristics:
    K = [[50, 15, 15, 12], [10, 6, 5, 4],
         [4, 3, 2, 1], [3, 2, 1, 0]]

    def __init__(self):
        pass

    @staticmethod
    def position_score(board: List[List[int]]):
        empty_cells = 0
        score = 0

        for x in range(4):
            for y in range(4):
                score += Heuristics.K[x][y] * board[x][y]
                if board[x][y] == 0:
                    empty_cells += 1

        return (score, empty_cells)

    @staticmethod
    def number_of_neighbors_score(board: List[List[int]]):
        def num_neighbors(row, col):
            num = 0
            if row - 1 >= 0:  # up
                num += 1
            if row + 1 <= 3:  # down
                num += 1
            if col - 1 >= 0:  # left
                num += 1
            if col + 1 <= 3:  # right
                num += 1
            return num

        score = 0
        for row in range(4):
            for col in range(4):
                number_of_neighbors = num_neighbors(row, col)
                # example: 512 * 1/4 is worse than 512 * 1/2
                score += (1.0/number_of_neighbors) * board[row][col]

        return ease_slope(score)

    @staticmethod
    def gradient_score(board: List[List[int]]):
        # it's worse if cell2 is greater, so this should be reflected
        # if cell1 is greater, then the best score is when cell1 and cell2 values are more similar
        def score_neighbors(cell1, cell2):
            # if they are the same, the score should reflect that this is good,
            # but not so good as to deter combining them.
            if cell1 == cell2:
                return 0
            diff = cell1 - cell2
            if cell1 > cell2:
                return 10/diff
            return diff

        score = 0

        for i in range(3):
            for j in range(4):
                if j < 3:
                    score_to_right = score_neighbors(
                        board[i][j], board[i][j + 1])
                    score += score_to_right
                score_down = score_neighbors(board[i][j], board[i + 1][j])
                score += score_down
        return score

    @staticmethod
    def corner_score(board: List[List[int]], maxTile: int):

        score: float = 0.0

        if board[0][0] == maxTile:
            score += maxTile
        else:
            score -= maxTile
        return score

    @staticmethod
    def rateBoard(weights: Dict[str, float], board: List[List[int]], maxTile: int, print_h=False):
        # rating = 0
        penalty = 0

        (cell_weights, blank_spaces) = Heuristics.position_score(board)

        if blank_spaces <= 1:
            penalty = -50

        # game legnth is used becasue something like cell_weights will become a much larger number over time
        # and you dont want this to overtake all the other factors
        # this function will increase from 0.5 -> 0.833 when maxTile is 1024.
        game_length = math.log(maxTile)/(1 + math.log(maxTile))

        K_spread = cell_weights*weights["A"]
        gradient_smoothness = Heuristics.gradient_score(board)*weights["B"]
        corner = Heuristics.corner_score(board, maxTile)*weights["C"]
        empty_cells = blank_spaces*weights["D"]*game_length
        penalty = penalty*weights["F"]*game_length

        if print_h:
            print(f"K_spread:    {K_spread}")
            print(f"gradient: {gradient_smoothness}")
            print(f"corner:      {corner}")
            print(f"empty_cells: {empty_cells}")
            print(f"penalty:     {penalty}")

        total_score = sum((
            K_spread,
            gradient_smoothness,
            corner,
            empty_cells,

            penalty
        ))
        # print(f'total: {total_score}')
        return total_score


def getNewTileValue():
    if randint(0, 99) < 100 * probability:
        return possibleNewTiles[0]
    else:
        return possibleNewTiles[1]


def get_child(depth, child_action, config):
    new_grid = Grid_State(depth, child_action)
    new_grid.map = [x[:] for x in config]
    return new_grid


class Grid_State(Grid):
    def __init__(self, depth, move, size=4):
        super().__init__()
        self.utility = 0
        self.children = []
        self.depth = depth
        self.action = move

    def expand(self, weights):
        if self.depth % 2 == 0:
            self.player_moves(weights)
        else:
            self.comp_moves(weights)

    def player_moves(self, weights):

        moves = self.getAvailableMoves()

        for choice in [0, 2, 3, 1]:
            if choice in moves:
                if self.canMove([choice]):
                    child = get_child(self.depth + 1, choice, self.map)
                    child.move(choice)
                    if child.map != self.map:
                        child.utility = Heuristics.rateBoard(weights,
                                                             child.map,
                                                             child.getMaxTile())
                        self.children.append(child)

    def comp_moves(self, weights):

        cells = [(x, y) for x in range(4)
                 for y in range(4) if self.map[x][y] == 0]

        for cell in cells:
            if self.canInsert(cell):

                for possible_value in [2, 4]:
                    child = get_child(self.depth + 1, self.action, self.map)
                    child.setCellValue(cell, possible_value)
                    # child.setCellValue(cell, getNewTileValue())
                    child.utility = Heuristics.rateBoard(weights,
                                                         child.map,
                                                         child.getMaxTile()
                                                         )
                    self.children.append(child)

    def to_s(self):
        string = ''
        for row in self.map:
            for val in row:
                string += str(val)
        return string

    def __eq__(self, other):
        return str(self.map) == str(other.map)


class PlayerAI(BaseAI):
    def __init__(self):
        self.A = 1.5  # cell weights
        self.B = 2.0  # gradient
        self.C = 2.0  # corner
        self.D = 10.0  # blank spaces
        self.E = 0.9  #
        self.F = 5.0  # penalty
        self.number_of_weights = 6
        self.depth_limit = 5

    def get_weights_dict(self):
        return {"A": self.A, "B": self.B, "C": self.C,
                "D": self.D, "E": self.E, "F": self.F}

    def getMove(self, grid):

        search_start = time.clock()
        # explored = deque()
        stack = deque()
        explored = set()
        begin_state = Grid_State(0, "Initial")

        weights = self.get_weights_dict()

        begin_state.map = [x[:] for x in grid.map]
        stack.append(begin_state)
        rep_s = begin_state.to_s()

        explored.add(rep_s)
        while stack:

            node = stack.popleft()
            node.expand(weights)

            # if there are not children,
            # this is very bad
            num_children = len(node.children)
            if num_children == 0:
                node.utility = -10000

            for y in range(num_children - 1, -1, -1):
                child = node.children[y]
                if child.depth < self.depth_limit:
                    rep_s = child.to_s()
                    if rep_s not in explored:
                        # if node.children[y] not in stack:
                        stack.append(node.children[y])
                        explored.add(rep_s)
            # if time.clock() - search_start >= time_limit:
            #     #break_time = True
            #     break

        alpha = -float("inf")
        beta = float("inf")
        best_move = None

        for child in begin_state.children:
            utility = minimize(child, alpha, beta)
            if utility > alpha:
                alpha = utility
                best_move = child.action

        return best_move

    def get_offspring(self, weights):
        children = []
        random_mutation = uniform(-l_rate, l_rate)
        for w in range(self.number_of_weights):

            copy_parent = list(weights)
            copy_parent[w] += random_mutation
            children.append(tuple(copy_parent))
        return children

    def weights_tuple(self):
        return (self.A, self.B, self.C, self.D, self.E, self.F)

    def set_weights(self, weights):
        self.A = weights[0]
        self.B = weights[1]
        self.C = weights[2]
        self.D = weights[3]
        self.E = weights[4]
        self.F = weights[5]


def maximize(node, alpha, beta):

    if not node.children:
        return node.utility

    maxUtility = -float("inf")

    for child in node.children:
        maxUtility = max(maxUtility, minimize(child, alpha, beta))

        if maxUtility >= beta:

            return maxUtility

        # alpha = max(alpha, maxUtility)
        if maxUtility >= alpha:
            alpha = maxUtility
        else:
            alpha = alpha

    return maxUtility


def minimize(node, alpha, beta):

    if not node.children:
        return node.utility

    minUtility = float("inf")
    # print(node.depth)
    for child in node.children:
        minUtility = min(minUtility, maximize(child, alpha, beta))

        if minUtility <= alpha:
            return minUtility

        # beta = min(beta, minUtility)
        if minUtility <= beta:
            beta = minUtility
        else:
            beta = beta

    return minUtility

# print(rateBoard([[512, 128, 8, 16], [8, 64, 32, 8], [2, 32, 8, 4], [32, 16, 4, 2]], 5, 4, 512, 1))
