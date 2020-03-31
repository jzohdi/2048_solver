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
mutation_degree = 2.0
# heuristic functions
#


def ease_slope(number):
    if number < 0:
        return -((-number)**(1/2))
    return number**(1/2)


def transform(cell: int):
    if cell > 0:
        return math.log2(cell)
    return 0


class Heuristics:
    # K = [[5, 4, 3, 2], [4, 3, 2, 1],
    #      [3, 2, 1, 0], [-2, -4, -6, -10]]
    K = [[20, 15, 15, 12], [13, 10, 6, 4],
         [4, 3, 2, 1], [3, 2, 1, 0]]

    def __init__(self):
        pass

    @staticmethod
    def position_score(board: List[List[int]], max_tile):
        # get no credit if property is not retained

        empty_cells = 0
        score = 0

        for x in range(4):
            for y in range(4):
                score += Heuristics.K[x][y] * transform(board[x][y])
                if board[x][y] == 0:
                    empty_cells += 1

        if empty_cells <= 1:
            empty_cells = -3

        return (ease_slope(score), empty_cells)

    @staticmethod
    def correct_position_score(board: List[List[int]]):
        pass

    # the closer to the top of the board, the better the value
    @staticmethod
    def rate_top_row(board: List[List[int]], max_tile, print_h=False):
        def get_diff(ref_cell, list_of_cells):
            diff = 0
            for cell in list_of_cells:
                diff -= (ref_cell - cell)
            return diff

        top_row = board[0]
        if print_h:
            print("top row: ")
            print(top_row)
        # if the corner is not the max_tile,
        # its still better to maximize the top row
        if top_row[0] != max_tile:
            return ease_slope(-max_tile + get_diff(max_tile, top_row))
        else:
            return ease_slope(sum(top_row))

    @staticmethod
    def gradient_score(board: List[List[int]]):
        # the close together the cells are, the better the score should be
        def score_neighbors(cell1, cell2):
            if cell1 == cell2:
                return 0
            diff = abs(cell1 - cell2)   # + 1 in case they are the same
            return 2/diff

        score = 0
        for i in range(3):
            for j in range(4):
                if j < 3:
                    score_to_right = score_neighbors(
                        board[i][j], board[i][j + 1])
                    score += score_to_right
                score_down = score_neighbors(board[i][j], board[i + 1][j])
                score += (score_down/3)
        return ease_slope(score)

    @staticmethod
    def top_row_full(top_row: List[int], max_tile):
        return not any([x == 0 for x in top_row]) and max_tile in top_row

    @staticmethod
    def find_top_vals(board: List[List[int]], number_of_vals: int):
        flattened_board = [cell for row in board for cell in row]
        flattened_board.sort()
        top_values = flattened_board[-number_of_vals:]
        return top_values

    @staticmethod
    def corner_score(board: List[List[int]], max_tile: int):

        score: float = 0.0

        if board[0][0] == max_tile:
            score += math.log2(max_tile)
        else:
            score -= math.log2(max_tile)
        return score

    @staticmethod
    def top_vals_penalty(board: List[List[int]]):
        penalty = 0
        top_values = Heuristics.find_top_vals(board, 4)
        rest_of_board = board[1:]
        flat_rest = [cell
                     for row in rest_of_board
                     for cell in row
                     ]
        for val in top_values:
            if val in flat_rest:
                penalty += val

        return -math.log2(penalty) if penalty > 0 else 0

    @staticmethod
    def rateBoard(weights: Tuple[float], board: List[List[int]], max_tile: int, move, print_h=False):

        # game legnth is used becasue something like cell_weights will become a much larger number over time
        # and you dont want this to overtake all the other factors
        # this function will increase from 0.5 -> 0.833 when max_tile is 1024.
        game_length = math.log(max_tile)/(1 + math.log(max_tile))

        bonus = 0
        penalty = 0

        (cell_weights, blank_spaces) = Heuristics.position_score(board, max_tile)

        if blank_spaces <= 1:
            penalty -= 2*math.log2(max_tile)
        if board[0][0] != max_tile:
            penalty -= 4*math.log2(max_tile)
        if move == 1:
            penalty -= math.log2(max_tile)

        K_spread = cell_weights*weights[0]
        gradient_smoothness = Heuristics.gradient_score(
            board)*weights[1]
        corner = Heuristics.corner_score(board, max_tile)*weights[2]
        empty_cells = blank_spaces*weights[3]*game_length

        if Heuristics.top_row_full(board[0], max_tile):
            bonus += 4 * math.log2(max_tile)
        # bonus = bonus*weights[5]
        top_vals_not_in_top_row = Heuristics.top_vals_penalty(
            board) * weights[4]

        if print_h:
            print(f"move: {move}")
            print(f"K_spread:    {K_spread}")
            print(f"gradient: {gradient_smoothness}")
            print(f"corner:      {corner}")
            print(f"empty_cells: {empty_cells}")
            # print(f"rows score: {row_rating}")
            print(f"top row penalty {top_vals_not_in_top_row}")
            print(f"bonus:     {bonus}")
            print(f"penalty: {penalty}")

        return sum((
            K_spread,
            gradient_smoothness,
            corner,
            empty_cells,
            # row_rating,
            top_vals_not_in_top_row,
            bonus,
            penalty
        ))


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
                                                             child.getMaxTile(),
                                                             choice)
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
                                                         child.getMaxTile(),
                                                         self.action)
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
        self.weights = (
            2.4,  # K_spread,
            5.0,  # gradient,
            1.2,  # corner,
            2.0,  # empties,
            2.0,  # row score,
            2.0  # bonus
        )

        self.number_of_weights = 6
        self.depth_limit = 6

    def getMove(self, grid):

        search_start = time.clock()
        # explored = deque()
        stack = deque()
        explored = set()
        begin_state = Grid_State(0, "Initial")

        weights = self.weights_tuple()

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
        random_mutation = uniform(-mutation_degree, mutation_degree)
        for w in range(self.number_of_weights):

            copy_parent = list(weights)
            copy_parent[w] += random_mutation
            children.append(tuple(copy_parent))
        return children

    def weights_tuple(self):
        return self.weights

    def set_weights(self, weights):
        self.weights = weights


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
