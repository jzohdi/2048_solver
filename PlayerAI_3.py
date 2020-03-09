# -*- coding: utf-8 -*-
"""
Created on Fri Jul  6 22:25:11 2018

@author: jake
"""

import time
import math
from Grid_3 import Grid
from random import randint
from BaseAI_3 import BaseAI
from collections import deque

K = [[20, 15, 15, 15], [10, 4, 3, 2], [4, 3, 2, 1], [3, 2, 1, 0]]
probability = 0.7
possibleNewTiles = [2, 4]
depth_limit = 6
time_limit = 2

# heuristic functions
#


def position_score(board):

    empty_cells = 0
    score = 0
    for x in range(4):
        for y in range(4):
            score += K[x][y] * board[x][y]
            if board[x][y] == 0:
                empty_cells += 3

    if empty_cells != 0:
        empty_cells = math.log(empty_cells)/math.log(2)

    return (score, empty_cells)


def row_likeness(board):

    score = 0
    for x in range(4):
        for y in range(3):
            score -= int(abs(board[x][y] - board[x][y+1]))

    return score


def corners(board, maxTile):

    score = 0

    if board[0][0] == maxTile:
        score += maxTile

    if score != 0:
        score = math.log(score)
    # if board[0][0] == 2048:
    #   score += 10
    return score


def rateBoard(board, size, maxTile, action):

    A = 2
    B = 1.2
    C = 2
    D = 3
    E = 0.20
    # rating = 0
    penalty = 0
    # weight of each cell
    (cell_weights, blank_spaces) = position_score(board)

    row_smooth = row_likeness(board)*math.log(maxTile)/E

    corner = corners(board, maxTile)

    if blank_spaces <= 2:
        penalty = 250

    game_length = math.log(maxTile)/math.log(4) + math.log(maxTile)

    return cell_weights*A*game_length + row_smooth*B * + corner*C + game_length*blank_spaces*D - penalty*game_length


states_visited = {}


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

    def expand(self):
        if self.depth % 2 == 0:
            self.player_moves()
        else:
            self.comp_moves()

#    def get_child(self, child_action):
#        start_time = time.time()
#        new_grid = Grid_State(self.depth + 1, child_action)
#        new_grid.map = [x[:] for x in self.map]
#        return new_grid

    def player_moves(self):

        moves = self.getAvailableMoves()

        for choice in [0, 2, 3, 1]:
            if choice in moves:
                if self.canMove([choice]):
                    child = get_child(self.depth + 1, choice, self.map)
                    child.move(choice)
                    if child.map != self.map:
                        child.utility = rateBoard(
                            child.map, child.size, child.getMaxTile(), child.action)
                        self.children.append(child)

    def comp_moves(self):

        cells = [(x, y) for x in range(4)
                 for y in range(4) if self.map[x][y] == 0]
        for cell in cells:
            if self.canInsert(cell):

                child = get_child(self.depth + 1, self.action, self.map)
                child.setCellValue(cell, getNewTileValue())
                child.utility = rateBoard(
                    child.map, child.size, child.getMaxTile(), child.action)
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

    def getMove(self, grid):

        search_start = time.clock()
        #explored = deque()
        stack = deque()
        explored = set()
        begin_state = Grid_State(0, "Initial")

        begin_state.map = [x[:] for x in grid.map]
        stack.append(begin_state)
        repr = begin_state.to_s()

        explored.add(repr)
        while stack:

            node = stack.popleft()
            node.expand()

            # for x in range(len(node.children)):
            #   print(node.children[x].map, node.children[x].utility, node.children[x].depth, node.children[x].action)
            for y in range(len(node.children) - 1, -1, -1):
                child = node.children[y]
                if child.depth < depth_limit:
                    repr = child.to_s()
                    if repr not in explored:
                        # if node.children[y] not in stack:
                        stack.append(node.children[y])
                        explored.add(repr)
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


def maximize(node, alpha, beta):

    if not node.children:
        return node.utility

    maxUtility = -float("inf")

    for child in node.children:
        maxUtility = max(maxUtility, minimize(child, alpha, beta))

        if maxUtility >= beta:
            return maxUtility

        #alpha = max(alpha, maxUtility)
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

        #beta = min(beta, minUtility)
        if minUtility <= beta:
            beta = minUtility
        else:
            beta = beta

    return minUtility

#print(rateBoard([[512, 128, 8, 16], [8, 64, 32, 8], [2, 32, 8, 4], [32, 16, 4, 2]], 5, 4, 512, 1))
