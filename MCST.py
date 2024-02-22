import copy
import os
import time
import uuid
from io import BytesIO
from pympler import asizeof
import numpy as np
import chess
import matplotlib.pyplot as plt
# import cairosvg
from stockfish import Stockfish
from pympler import asizeof

stockfish = Stockfish(path=r"C:\Users\Mateu\Desktop\stockfish.exe")
stockfish.set_turn_perspective(True)
t = time.time()
print(stockfish.get_perft(1))
print(time.time() - t)

board = chess.Board()
t = time.time()
print(list(board.legal_moves))
print(time.time() - t)

class Node:
    def __init__(self, board: str, move: str, parent: "Node" = None):
        if parent is not None:
            stockfish.set_fen_position(board)
            stockfish.make_moves_from_current_position([move])

        self.visits = 0
        self.board = stockfish.get_fen_position()
        self.legal_moves = stockfish.get_perft(1)[1].keys()
        self.children = []
        self.parent = parent
        self.evaluation = self.get_evaluation()

    def is_terminal(self):
        self.visits += 1
        return len(self.children) == 0

    def is_fully_expanded(self):
        return len(self.children) == len(self.legal_moves)

    def get_untried_actions(self):
        actions = []
        for move in self.legal_moves:
            actions.append(Node(self.board, move, self))
        return actions

    def get_evaluation(self):
        eval = stockfish.get_static_eval()

        if eval is None:
            if len(self.legal_moves) == 0:
                return -9999
            else:
                eval = self.parent.evaluation

        return eval

class MTST:
    def __init__(self, root, budget):
        self.root = root
        self.budget = budget
        self.c = 1.4

    def run(self):
        for iter in range(self.budget):
            node = self.tree_policy(self.root)
            evaluation = self.default_policy(node)
            self.backpropagate(node, evaluation)
        return self.best_child(self.root)

    def tree_policy(self, node):
        while not node.is_terminal():
            if not node.is_fully_expanded():
                return self.expand(node)
            else:
                node = self.best_child(node)
        return node

    def expand(self, node):
        untried_actions = node.legal_moves
        random_child = np.random.choice(untried_actions)
        return node.children.append(random_child)

    def best_child(self, node):
        uct_values = [self.UCT(child) for child in node.children]
        return node.children[np.argmax(uct_values)]

    def default_policy(self, node):
        while not node.is_terminal():
            a = np.random.choice(node.get_untried_actions())
            node = node.children.append(a)
        return node.evaluation

    def backpropagate(self, node, evaluation):
        while node is not None:
            node.evaluation = evaluation
            node = node.parent

    def UCT(self, node):
        return (
                node.evaluation / node.visits
                + self.c
                + np.sqrt(np.log(node.parent.visits) / node.visits)
        )


stockfish = Stockfish(path=r"C:\Users\Mateu\Desktop\stockfish.exe")

search_tree = MTST(Node(stockfish.get_fen_position(), "", None), 5)
search_tree.run()
