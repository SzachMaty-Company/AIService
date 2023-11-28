import copy
import os
import time
import uuid
from io import BytesIO

import numpy as np
import chess
import matplotlib.pyplot as plt
import chess.svg
import cairosvg


def process_board(board):
    board = board.board_fen().split("/")
    board = ["0" * int(y) if y.isnumeric() else y for x in board for y in x]
    board = [item for sublist in board for item in sublist]
    return np.array(board).reshape(8, 8)


board = chess.Board()
board.push_san("e4")
board_svg = chess.svg.board(board=board)


def show_board(board):
    img_path = str(uuid.uuid4())

    with open(img_path, "wb") as img_file:
        img_file.write(board_svg.encode("utf-8"))

    img_data = cairosvg.svg2png(url=img_path)
    img = plt.imread(BytesIO(img_data))
    plt.imshow(img)
    plt.axis("off")
    plt.show()
    os.remove(img_path)


for move in board.legal_moves:
    print(move.uci())
    tmp = copy.deepcopy(board)
    tmp.push(move)
    # show_board(tmp)
    print(tmp, end="\n\n")

# print(board)
exit()


class Node:
    def __init__(self, board):
        self.visits = 0
        self.board = board
        self.children = []
        self.parent = None
        self.result = None
        self.evaluation = self.get_evaluation()

    def is_terminal(self):
        self.visits += 1
        return len(self.children) == 0

    def is_fully_expanded(self):
        return len(self.children) == len(self.board.legal_moves)

    def get_untried_actions(self):
        actions = []
        for move in self.board.legal_moves:
            tmp = copy.copy(self.board)
            tmp.push_san(move.uci())
            actions.append(Node(None, tmp))
        return actions

    def get_evaluation(self):
        if self.board.is_checkmate():
            return -1000
        if self.board.is_stalemate():
            return 0

        evaluation = 0
        mapper = {
            "P": 1,
            "N": 3,
            "B": 3,
            "R": 5,
            "Q": 9,
            "K": 50,
            "p": -1,
            "n": -3,
            "b": -3,
            "r": -5,
            "q": -9,
            "k": -50,
        }

        for x in self.board.board_fen():
            if x in mapper:
                evaluation += mapper[x]
        return evaluation


class MTST:
    def __init__(self, root, budget):
        self.root = root
        self.budget = budget
        self.c = 1.4

    def run(self):
        for iter in range(self.budget):
            node = self.tree_policy(self.root)
            result = self.default_policy(node)
            self.backpropagate(node, result)
        return self.best_child(self.root)

    def tree_policy(self, node):
        while not node.is_terminal():
            if not node.is_fully_expanded():
                return self.expand(node)
            else:
                node = self.best_child(node)
        return node

    def expand(self, node):
        untried_actions = node.get_untried_actions()
        random_child = np.random.choice(untried_actions)
        return node.children.append(random_child)

    def best_child(self, node):
        uct_values = [self.UCT(child) for child in node.children]
        return node.children[np.argmax(uct_values)]

    def default_policy(self, node):
        while not node.is_terminal():
            a = np.random.choice(node.get_untried_actions())
            node = node.children.append(a)
        return node.result

    def backpropagate(self, node, result):
        while node is not None:
            node.result = result
            node = node.parent

    def UCT(self, node):
        return node.evaluation / node.visits + self.c + np.sqrt(np.log(node.parent.visits) / node.visits)
