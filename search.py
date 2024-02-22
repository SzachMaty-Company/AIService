import copy

import numpy as np
import math
import chess

# https://github.com/foersterrobert/AlphaZeroFromScratch/blob/main


class Node:
    def __init__(self, board,c,   player=1, parent=None, action_taken=None):
        self.board = board
        self.c = c
        self.parent = parent
        self.action_taken = action_taken
        self.evaluation = self.get_evaluation()
        self.children = []
        self.expandable_moves = self.get_untried_actions()
        self.player = player
        self.visit_count = 0
        self.value_sum = 0

    def get_untried_actions(self):
        actions = []
        for move in self.board.legal_moves:
            actions.append(move.uci())
        return actions

    def get_evaluation(self):
        evaluation = 0
        mapper = {
            "P": 100,
            "N": 300,
            "B": 300,
            "R": 500,
            "Q": 900,
            "K": 5000,
            "p": -100,
            "n": -300,
            "b": -300,
            "r": -500,
            "q": -900,
            "k": -5000,
        }

        for x in self.board.board_fen():
            if x in mapper:
                evaluation += mapper[x]
        return evaluation

    def get_value_and_terminated(self, action):
        if self.check_win(action):
            return 1, True
        if len(self.get_untried_actions()) == 0:
            return 0, True
        return self.get_evaluation(), False

    def check_win(self, action):
        if action is None:
            return False
        return self.board.is_checkmate()

    def is_fully_expanded(self):
        return len(self.expandable_moves) == 0 and len(self.children) > 0

    def select(self):
        best_child = None
        best_ucb = -np.inf

        for child in self.children:
            ucb = self.get_ucb(child)
            if ucb > best_ucb:
                best_child = child
                best_ucb = ucb

        return best_child

    def get_ucb(self, child):
        q_value = 1 - ((child.value_sum / child.visit_count) + 1) / 2
        return q_value + self.c * math.sqrt(math.log(self.visit_count) / child.visit_count)

    def expand(self):

        action = np.random.choice(self.expandable_moves[0])
        self.expandable_moves.remove(action)

        child_state = copy.copy(self.board)
        child_state.push_san(action)
        child = Node(board=child_state, c=self.c, player=-self.player, parent=self, action_taken=action)

        self.children.append(child)
        return child

    def backpropagate(self, value):
        self.value_sum += value
        self.visit_count += 1

        value = -value
        if self.parent is not None:
            self.parent.backpropagate(value)


class MCTS:
    def __init__(self, node, args):
        self.root = node
        self.args = args

    def search(self, state):
        root = Node(board=state, c=self.args["C"])

        for search in range(self.args['num_searches']):
            node = root

            while node.is_fully_expanded():
                node = node.select()

            value, is_terminal = self.root.get_value_and_terminated(node.state, node.action_taken)
            value = -value

            if not is_terminal:
                node = node.expand()
                value = node.simulate()

            node.backpropagate(value)

        action_probs = np.zeros(self.root.action_size)
        for child in root.children:
            action_probs[child.action_taken] = child.visit_count
        action_probs /= np.sum(action_probs)
        return action_probs


player = 1

args = {
    'C': 1.41,
    'num_searches': 1000
}

mcts = MCTS(Node, args)

state = chess.Board()

while True:
    print(state)

    if player == 1:
        valid_moves = state.get_valid_moves(state)
        print("valid_moves", [i for i in range(state.action_size) if valid_moves[i] == 1])
        action = int(input(f"{player}:"))

        if valid_moves[action] == 0:
            print("action not valid")
            continue

    else:
        neutral_state = state.change_perspective(state, player)
        mcts_probs = mcts.search(neutral_state)
        action = np.argmax(mcts_probs)

    state = state.get_next_state(state, action, player)

    value, is_terminal = state.get_value_and_terminated(state, action)

    if is_terminal:
        print(state)
        if value == 1:
            print(player, "won")
        else:
            print("draw")
        break

    player = state.get_opponent(player)