import collections
import copy
import math
import random
import time

import chess_board as chess
import numpy as np


class CustomBoard(chess.Board):
    pass
np.seterr(divide='ignore')


class UCTNode:
    def __init__(self, board, move, move_index,parent=None, depth=0):
        self.board = board
        self.move_index = move_index
        self.child_move_index = 0
        self.move = move
        self.legal_moves = board.generate_legal_moves()
        self.is_expanded = False
        self.parent = parent
        self.children = [None]*200
        self.child_total_value = np.zeros([200], dtype=np.float32)
        self.child_number_visits = np.repeat(0.000001, 200)
        self.depth = depth
    @property
    def number_visits(self):
        return self.parent.child_number_visits[self.move_index]

    @number_visits.setter
    def number_visits(self, value):
        self.parent.child_number_visits[self.move_index] = value

    @property
    def total_value(self):
        return self.parent.child_total_value[self.move_index]

    @total_value.setter
    def total_value(self, value):
        self.parent.child_total_value[self.move_index] = value

    def child_Q(self):
        return self.child_total_value / self.child_number_visits

    def get_most_visited(self) -> tuple:
        most_visited_index = np.argmax(self.child_number_visits)

        return self.children[most_visited_index].move

    def child_U(self):
        return np.sqrt(np.log(self.number_visits) / (1+self.child_number_visits))# or 0.0000001))

    def value(self, root_color):
        return (self.child_Q() )* (1 if self.board.turn != root_color else -1) + 1.5 * self.child_U()

    def best_child(self, root_color):
        best = None

        best_index = np.argmax(self.child_Q() + 1.5*self.child_U())
        best = self.children[best_index]

        return best

    def select_leaf(self, root_color):
        current = self
        while current.is_expanded:
            best = current.best_child(root_color)
            if best:
                current = best
            else:
                break
        return current

    def expand(self):
        # print(*self.legal_moves)
        try:
            move = next(self.legal_moves)
            copy_board = self.board.copy(stack=False)
            copy_board.push(move)
            self.children[self.child_move_index] = UCTNode(copy_board, move, self.child_move_index,parent=self, depth=self.depth + 1)
            self.child_move_index += 1
        except StopIteration:
            self.is_expanded = True

    def backup(self, root_color):
        self.number_visits += 1
        self.total_value = eval(self.board, root_color)

        current = self.parent
        while current.parent is not None:
            current.number_visits += 1

            if current.board.turn == root_color:
                current.total_value += np.sum(current.child_total_value)
            else:
                current.total_value -= np.sum(current.child_total_value)

            current = current.parent


class DummyNode(object):
    def __init__(self):
        self.parent = None
        self.child_total_value = collections.defaultdict(float)
        self.child_number_visits = collections.defaultdict(float)


def UCT_search(game_state, num_reads):
    root = UCTNode(game_state,move_index=0 ,move=None, parent=DummyNode())
    for i in range(num_reads):
        leaf = root.select_leaf(root.board.turn)

        if leaf.board.is_checkmate():
            if leaf.board.turn == root.board.turn:
                leaf.total_value = -10000
            else:
                leaf.total_value = 10000

        leaf.expand()
        leaf.backup(root_color=root.board.turn)
    print(root.child_total_value)
    print(root.child_number_visits)
    print("=====================================")
    # while True:
    #     print(root.child_total_value)
    #     print(root.child_number_visits)
    #     print(chess.Move(*root.children[root.get_most_visited()].move), root.children[root.get_most_visited()].move)
    #     print("=====================================")
    #     root = root.children[root.get_most_visited()]

    return root.get_most_visited()


PAWN = 100
KNIGHT = 320
BISHOP = 330
ROOK = 500
QUEEN = 900
KING = 20000
PAWN_TABLE = [
    0, 0, 0, 0, 0, 0, 0, 0,
    50, 50, 50, 50, 50, 50, 50, 50,
    10, 10, 20, 30, 30, 20, 10, 10,
    5, 5, 10, 25, 25, 10, 5, 5,
    0, 0, 0, 20, 20, 0, 0, 0,
    5, -5, -10, 0, 0, -10, -5, 5,
    5, 10, 10, -20, -20, 10, 10, 5,
    0, 0, 0, 0, 0, 0, 0, 0
]
PAWN_TABLE_BLACK = PAWN_TABLE[::-1]
KNIGHT_TABLE = [
    -50, -40, -30, -30, -30, -30, -40, -50,
    -40, -20, 0, 0, 0, 0, -20, -40,
    -30, 0, 10, 15, 15, 10, 0, -30,
    -30, 5, 15, 20, 20, 15, 5, -30,
    -30, 0, 15, 20, 20, 15, 0, -30,
    -30, 5, 10, 15, 15, 10, 5, -30,
    -40, -20, 0, 5, 5, 0, -20, -40,
    -50, -40, -30, -30, -30, -30, -40, -50
]
KNIGHT_TABLE_BLACK = KNIGHT_TABLE[::-1]
BISHOP_TABLE = [
    -20, -10, -10, -10, -10, -10, -10, -20,
    -10, 0, 0, 0, 0, 0, 0, -10,
    -10, 0, 5, 10, 10, 5, 0, -10,
    -10, 5, 5, 10, 10, 5, 5, -10,
    -10, 0, 10, 10, 10, 10, 0, -10,
    -10, 10, 10, 10, 10, 10, 10, -10,
    -10, 5, 0, 0, 0, 0, 5, -10,
    -20, -10, -10, -10, -10, -10, -10, -20
]
BISHOP_TABLE_BLACK = BISHOP_TABLE[::-1]
ROOK_TABLE = [
    0, 0, 0, 0, 0, 0, 0, 0,
    5, 10, 10, 10, 10, 10, 10, 5,
    -5, 0, 0, 0, 0, 0, 0, -5,
    -5, 0, 0, 0, 0, 0, 0, -5,
    -5, 0, 0, 0, 0, 0, 0, -5,
    -5, 0, 0, 0, 0, 0, 0, -5,
    -5, 0, 0, 0, 0, 0, 0, -5,
    0, 0, 0, 5, 5, 0, 0, 0
]
ROOK_TABLE_BLACK = ROOK_TABLE[::-1]
QUEEN_TABLE = [
    -20, -10, -10, -5, -5, -10, -10, -20,
    -10, 0, 0, 0, 0, 0, 0, -10,
    -10, 0, 5, 5, 5, 5, 0, -10,
    -5, 0, 5, 5, 5, 5, 0, -5,
    0, 0, 5, 5, 5, 5, 0, -5,
    -10, 5, 5, 5, 5, 5, 0, -10,
    -10, 0, 5, 0, 0, 0, 0, -10,
    -20, -10, -10, -5, -5, -10, -10, -20
]
QUEEN_TABLE_BLACK = QUEEN_TABLE[::-1]
KING_TABLE = [
    -30, -40, -40, -50, -50, -40, -40, -30,
    -30, -40, -40, -50, -50, -40, -40, -30,
    -30, -40, -40, -50, -50, -40, -40, -30,
    -30, -40, -40, -50, -50, -40, -40, -30,
    -20, -30, -30, -40, -40, -30, -30, -20,
    -10, -20, -20, -20, -20, -20, -20, -10,
    20, 20, 0, 0, 0, 0, 20, 20,
    20, 30, 10, 0, 0, 10, 30, 20
]
KING_TABLE_BLACK = KING_TABLE[::-1]
KING_TABLE_END = [
    -50, -40, -30, -20, -20, -30, -40, -50,
    -30, -20, -10, 0, 0, -10, -20, -30,
    -30, -10, 20, 30, 30, 20, -10, -30,
    -30, -10, 30, 40, 40, 30, -10, -30,
    -30, -10, 30, 40, 40, 30, -10, -30,
    -30, -10, 20, 30, 30, 20, -10, -30,
    -30, -30, 0, 0, 0, 0, -30, -30,
    -50, -30, -30, -30, -30, -30, -30, -50
]
KING_TABLE_END_BLACK = KING_TABLE_END[::-1]


def eval(board: chess.Board, root_color):
    white_eval = 0
    black_eval = 0
    is_endgame = False

    if board.queens == 0:
        is_endgame = True
    if board.queens & board.occupied_co[chess.WHITE] == 0 and board.queens & board.occupied_co[chess.BLACK] != 0:
        is_endgame = count_ones((board.knights | board.bishops | board.rooks) & board.occupied_co[chess.WHITE]) <= 1
    elif board.queens & board.occupied_co[chess.BLACK] == 0 and board.queens & board.occupied_co[chess.WHITE] != 0:
        is_endgame = count_ones((board.knights | board.bishops | board.rooks) & board.occupied_co[chess.BLACK]) <= 1

    for figure, weight, table in zip(
            [board.pawns, board.knights, board.bishops, board.rooks, board.queens, board.kings],
            [PAWN, KNIGHT, BISHOP, ROOK, QUEEN, KING],
            [[PAWN_TABLE, PAWN_TABLE_BLACK], [KNIGHT_TABLE, KNIGHT_TABLE_BLACK], [BISHOP_TABLE, BISHOP_TABLE_BLACK],
             [ROOK_TABLE, ROOK_TABLE_BLACK], [QUEEN_TABLE, QUEEN_TABLE_BLACK],
             [KING_TABLE if not is_endgame else KING_TABLE_END,
              KING_TABLE_BLACK if not is_endgame else KING_TABLE_END_BLACK]]):
        white_eval += count_ones(board.occupied_co[chess.WHITE] & figure) * weight + sum(
            table[chess.WHITE][i] for i in range(64) if board.occupied_co[chess.WHITE] & figure & (1 << i))
        black_eval += count_ones(board.occupied_co[chess.BLACK] & figure) * weight + sum(
            table[chess.BLACK][i] for i in range(64) if board.occupied_co[chess.BLACK] & figure & (1 << i))

    return (white_eval - black_eval) if root_color else (black_eval - white_eval)


def count_ones(byte):
    count = 0
    while byte:
        count += byte & 1
        byte >>= 1
    return count


def print_byte(byte):
    if type(byte) == list:
        for b in byte:
            string = format(b, '0{}b'.format(64))
            for i in range(0, len(string), 8):
                print(string[i:i + 8])
            print("\n")
    else:
        string = format(byte, '0{}b'.format(64))
        for i in range(0, len(string), 8):
            print(string[i:i + 8])
    print("\n")


def MCTS_self_play(num_games):
    for idxx in range(0, num_games):
        current_board = chess.Board("r1bqkb1r/ppp1pppp/2n2n2/3p4/3P1B2/5N1P/PPP1PPP1/RN1QKB1R b KQkq - 0 4")
        states = []
        while current_board.fullmove_number <= 100:
            draw_counter = 0
            for s in states:
                if np.array_equal(current_board.board_fen(), s):
                    draw_counter += 1
            if draw_counter == 3:
                break
            states.append(copy.deepcopy(current_board.board_fen()))
            t = time.time()
            best_move = UCT_search(current_board, 30000)
            print("time: ", time.time() - t)
            current_board.push(best_move)
            print(current_board, eval(current_board, not current_board.turn), best_move)

            if current_board.is_checkmate():
                if current_board.turn:  # black wins
                    print("Black wins")
                else:  # white wins
                    print("White wins")
                break


MCTS_self_play(50)
