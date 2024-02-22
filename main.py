import random
from math import log, sqrt

import chess
from stockfish import Stockfish


# https://github.com/ImparaAI/monte-carlo-tree-search

class Node:

    def __init__(self, state: str, move: str| None, parent: "Node"| None = None):
        if move:
            stockfish.set_fen_position(state)
            stockfish.make_moves_from_current_position([move])
        self.state = state
        self.win_value = 0
        self.policy_value = None
        self.visits = 0
        self.parent = parent
        self.children = []
        self.legal_moves = stockfish.get_perft(1)[1].keys()
        self.expanded = False
        self.player_number = None
        self.discovery_factor = 0.35
        self.eval = self.get_evaluation()

    def update_win_value(self, value):
        self.win_value += value
        self.visits += 1

        if self.parent:
            self.parent.update_win_value(value)

    def update_policy_value(self, value):
        self.policy_value = value

    def add_child(self, move):
        child = Node(self.state, move, self)
        self.children.append(child)
        return child

    def add_children(self, children):
        for child in children:
            self.add_child(child)

    def get_preferred_child(self, root_node):
        best_children = []
        best_score = float('-inf')

        for child in self.children:
            score = child.get_score(root_node)

            if score > best_score:
                best_score = score
                best_children = [child]
            elif score == best_score:
                best_children.append(child)

        return random.choice(best_children)

    def get_score(self, root_node):
        discovery_operand = self.discovery_factor * (self.policy_value or 1) * sqrt(
            log(self.parent.visits) / (self.visits or 1))

        win_multiplier = 1 if self.parent.player_number == root_node.player_number else -1
        win_operand = win_multiplier * self.win_value / (self.visits or 1)

        self.score = win_operand + discovery_operand

        return self.score

    def is_scorable(self):
        return self.visits or self.policy_value != None


    def get_evaluation(self):
        eval = stockfish.get_static_eval()

        if eval is None:
            if len(self.legal_moves) == 0:
                return -9999
            else:
                eval = self.parent.evaluation

        return eval

class MonteCarlo:

    def __init__(self, root_node):
        self.root_node = root_node

    def make_choice(self):
        best_children = []
        most_visits = float('-inf')

        for child in self.root_node.children:
            if child.visits > most_visits:
                most_visits = child.visits
                best_children = [child]
            elif child.visits == most_visits:
                best_children.append(child)

        return random.choice(best_children)

    def make_exploratory_choice(self):
        children_visits = map(lambda child: child.visits, self.root_node.children)
        children_visit_probabilities = [visit / self.root_node.visits for visit in children_visits]
        random_probability = random.uniform(0, 1)
        probabilities_already_counted = 0.

        for i, probability in enumerate(children_visit_probabilities):
            if probabilities_already_counted + probability >= random_probability:
                return self.root_node.children[i]

            probabilities_already_counted += probability

    def simulate(self, expansion_count=1):
        for i in range(expansion_count):
            current_node = self.root_node

            while current_node.expanded:
                current_node = current_node.get_preferred_child(self.root_node)

            self.expand(current_node)

    def expand(self, node):
        for child in node.children:
            child_win_value = child.eval

            if child_win_value != None:
                child.update_win_value(child_win_value)

            if not child.is_scorable():
                self.random_rollout(child)
                child.children = []

        if len(node.children):
            node.expanded = True

    def random_rollout(self, node):
        move = random.choice(node.legal_moves)
        child = node.add_child(move)
        node.legal_moves.remove(move)
        child_win_value = child.eval

        if child_win_value != None:
            node.update_win_value(child_win_value)
        else:
            self.random_rollout(child)

stockfish = Stockfish(path=r"C:\Users\Mateu\Desktop\stockfish.exe")
stockfish.set_turn_perspective(True)

montecarlo = MonteCarlo(Node(stockfish.get_fen_position(), None))
montecarlo.simulate(10)
