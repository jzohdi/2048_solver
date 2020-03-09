# -*- coding: utf-8 -*-
"""
Created on Sat Jul 21 17:46:15 2018

@author: jake
"""

import time
import math
#from Grid_3 import Grid
from random import randint
from BaseAI_3 import BaseAI
from collections import deque

K = [[20, 15, 15, 15], [10, 6, 3, 2], [4, 3, 2, 1], [3, 2, 1, 0]]
probability = 0.8
possibleNewTiles = [2, 4]
depth_limit = 5
# time_limit = 0.5
vecIndex = [UP, DOWN, LEFT, RIGHT] = range(4)
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

    A = 0.8
    B = 0.9
    C = 2
    D = 3
    E = 0.29
    # rating = 0
    penalty = 0
    # weight of each cell
    (cell_weights, blank_spaces) = position_score(board)

    row_smooth = row_likeness(board)*math.log(maxTile)/E

    corner = corners(board, maxTile)

    if blank_spaces <= 3:
        penalty = 250

    game_length = math.log(maxTile)/math.log(4) + math.log(maxTile)

    return cell_weights*A*game_length + row_smooth*B*game_length + corner*C + game_length*blank_spaces*D - penalty*game_length


states = {}


def getNewTileValue():
    if randint(0, 99) < 100 * probability:
        return possibleNewTiles[0]
    else:
        return possibleNewTiles[1]


def get_child(depth, child_action, config):
    return Grid_State(depth, child_action, config)
    #key = str(config)
    # if key in states:
    #    return states[key]
    # else:
    #    new_grid = Grid_State(depth, child_action, config)
    #    #new_grid.map = config
    #    states[key] = new_grid
    #    return new_grid


def move(dir, grid):
    dir = int(dir)

    if dir == UP:
        return moveUD(False, grid)
    if dir == DOWN:
        return moveUD(True, grid)
    if dir == LEFT:
        return moveLR(False, grid)
    if dir == RIGHT:
        return moveLR(True, grid)

# Move Up or Down


def moveUD(down, grid):
    r = range(4 - 1, -1, -1) if down else range(4)

    moved = False

    for j in range(4):
        cells = []
        for i in r:
            cell = grid[i][j]

            if cell != 0:
                cells.append(cell)
        merge(cells)

        for i in r:
            value = cells.pop(0) if cells else 0
            if grid[i][j] != value:
                moved = True
            grid[i][j] = value

    return (moved, grid)
# move left or right


def moveLR(right, grid):
    r = range(4 - 1, -1, -1) if right else range(4)

    moved = False

    for i in range(4):
        cells = []
        for j in r:
            cell = grid[i][j]
            if cell != 0:
                cells.append(cell)
        merge(cells)

        for j in r:
            value = cells.pop(0) if cells else 0
            if grid[i][j] != value:
                moved = True
            grid[i][j] = value

    return (moved, grid)

# Merge Tiles


def merge(cells):
    if len(cells) <= 1:
        return cells
    i = 0
    while i < len(cells) - 1:
        if cells[i] == cells[i+1]:
            cells[i] *= 2

            del cells[i+1]
        i += 1


class Grid_State:
    def __init__(self, depth, move, config):
        # super().__init__()
        self.map = config
        self.utility = rateBoard(config, 4, max(
            [max(x) for x in config]), move)
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
        #moves = self.getAvailableMoves()
        for choice in [0, 2, 3, 1]:
            # if choice in moves:
            # if self.canMove([choice]):
            (can_move, new_grid) = move(choice, [x[:] for x in self.map])
            if can_move:
                child = get_child(self.depth + 1, choice, new_grid)
                # child.move(choice)
                if child.map != self.map:
                    #child.utility = rateBoard(child.map, child.size, max([max(x) for x in child.map]), child.action)
                    self.children.append(child)

    def comp_moves(self):

        cells = [(x, y) for x in range(4)
                 for y in range(4) if self.map[x][y] == 0]
        for cell in cells:
            if self.map[cell[0]][cell[1]] == 0:
                new_grid = [x[:] for x in self.map]
                new_grid[cell[0]][cell[1]] = getNewTileValue()
                child = get_child(self.depth + 1, self.action, new_grid)
                #child.map[cell[0]][cell[1]] = getNewTileValue()
                #child.setCellValue(cell, getNewTileValue())
                #child.utility = rateBoard(child.map, child.size, max([max(x) for x in child.map]), child.action)
                self.children.append(child)

    def to_s(self):
        rep = ''
        for row in self.map:
            for val in row:
                rep += str(val)
        return rep

    def __eq__(self, other):
        return str(self.map) == str(other.map)


class PlayerAI(BaseAI):

    def getMove(self, grid):

        search_start = time.clock()
        #explored = deque()
        explored = {}
        stack = deque()

        begin_state = Grid_State(0, "Initial", [x[:] for x in grid.map])

        stack.append(begin_state)

        while stack:

            node = stack.popleft()
            node.expand()

            # for x in range(len(node.children)):
            #   print(node.children[x].map, node.children[x].utility, node.children[x].depth, node.children[x].action)
            for child in node.children:

                # if node.children[y].depth < depth_limit:
                # if node.children[y] not in stack:
                # rep = child.to_s()
                if child.depth < depth_limit:  # and rep not in explored:
                    stack.append(child)
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
        #maxUtility = max(maxUtility, minimize(child, alpha, beta))
        utility = minimize(child, alpha, beta)

        if utility > maxUtility:
            maxUtility = utility
        else:
            maxUtility = maxUtility

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
        #minUtility = min(minUtility, maximize(child, alpha, beta))
        utility = maximize(child, alpha, beta)

        if utility < minUtility:
            minUtility = utility
        else:
            minUtility = minUtility

        if minUtility <= alpha:
            return minUtility

        #beta = min(beta, minUtility)
        if minUtility <= beta:
            beta = minUtility
        else:
            beta = beta

    return minUtility

#l = [[0, 2, 2, 0],[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]
#(can_move, new_grid) = move(2, [x[:] for x in l])
# if can_move:
#    print(new_grid, l)
#print(rateBoard([[512, 128, 8, 16], [8, 64, 32, 8], [2, 32, 8, 4], [32, 16, 4, 2]], 5, 4, 512, 1))
