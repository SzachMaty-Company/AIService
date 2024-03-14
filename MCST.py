import copy
import time

from chess.polyglot import zobrist_hash

import chess_board as chess
import numpy as np

np.seterr(divide='ignore')
MX = 0
trans = {}
class UCTNode:
    def __init__(self, board, move, move_index,parent=None, depth=0, alpha=float('-inf'), beta=float('inf')):
        self.board = board
        self.move_index = move_index
        self.child_move_index = 0
        self.move = move
        self.legal_moves = board.generate_legal_moves()
        self.is_expanded = False
        self.parent = parent
        self.children = []
        self.child_total_value = np.zeros([200], dtype=np.float32)
        self.child_number_visits = np.repeat(0, 200)
        self.depth = depth
        self.beta  = beta
        self.alpha = alpha
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


    def get_highest_value(self) -> tuple:
        most_visited_index = np.argmax(self.child_total_value)

        return self.children[most_visited_index].move


    def select_leaf(self, root_color):
        current = self

        while not current.is_expanded and current.depth < 4:
            best = current.expand(root_color)
            if type(best) == DummyNode:
                return current
            # best = current.children[current.child_move_index]

            if best.depth == 4:
                value = eval(best.board, root_color)
                best.total_value = value
                best.parent.alpha = max(best.parent.alpha, value)

                if best.parent.alpha >= best.parent.beta:
                    # print("pruning")
                    best.parent.is_expanded = True
                    best.parent.child_total_value = best.parent.child_total_value[:best.parent.child_move_index]
                    best.parent.child_number_visits = best.parent.child_number_visits[:best.parent.child_move_index]
                    best.parent.children = best.parent.children[:best.parent.child_move_index]
                    best.parent.backup(root_color)
                    return best.parent

            current = best

        return current.parent

    def expand(self, root_color):
        try:
            move = next(self.legal_moves)
            copy_board = self.board.copy(stack=False)
            copy_board.push(move)
            self.children.append(UCTNode(copy_board, move, self.child_move_index,parent=self, depth=self.depth + 1, alpha=self.alpha, beta=self.beta))
            self.child_move_index += 1
            return self.children[-1]
        except StopIteration:
            self.is_expanded = True
            self.child_total_value = self.child_total_value[:self.child_move_index]
            self.child_number_visits = self.child_number_visits[:self.child_move_index]
            self.children = self.children[:self.child_move_index]
            self.backup(root_color)
            return self.parent
    def backup(self, root_color):
        # self.total_value = eval(self.board, root_color)
        # self.number_visits += 1

        current = self
        # while current.parent is not None:
        current.number_visits += 1
        if current.is_expanded:
            if current.board.turn == root_color:
                if self.child_move_index == 0:
                    current.total_value = -10000
                else:
                    current.total_value = np.max(current.child_total_value)
            else:
                if self.child_move_index == 0:
                    current.total_value = 10000
                else:
                    current.total_value = np.min(current.child_total_value)
            current.parent.beta = min(current.parent.beta, current.total_value)
                # current = current.parent
            # else:
            #     break
    def __str__(self):
        return str(self.depth)


class DummyNode(object):
    def __init__(self):
        self.parent = None
        self.child_total_value = np.zeros([1], dtype=np.float32)
        self.child_number_visits = np.repeat(0.000001, 1)
        self.alpha = float('-inf')
        self.beta = float('inf')


def UCT_search(game_state, num_reads):
    root = UCTNode(game_state,move_index=0 ,move=None, parent=DummyNode())
    # for i in range(num_reads):
    leaf = root
    while True:
        leaf = leaf.select_leaf(root.board.turn)
        if leaf.board.is_checkmate():
            if leaf.board.turn == root.board.turn:
                leaf.total_value = -10000
            else:
                leaf.total_value = 10000
        if leaf.depth == 0:
            break
    # while True:
    #     print(np.array([(x, y, z,v,b) for x, y, z,v,b in zip(root.child_total_value, np.array(
    #         [str(chess.Move(*c.move)) if c else None for c in root.children]), root.child_number_visits, [x.beta for x in root.children], [x.alpha for x in root.children])]))
    #     most_visited_index = np.argmax(root.child_total_value)
    #     print(chess.Move(*root.children[most_visited_index].move), most_visited_index)
    #     print("=====================================")
    #     root = root.children[most_visited_index]

        # leaf = leaf.expand()
        # if leaf.depth == 3:
        #     leaf.backup(root_color=root.board.turn)

    return root.get_highest_value()


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
][::-1]
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
][::-1]
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
][::-1]
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
][::-1]
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
][::-1]
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
][::-1]
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
][::-1]
KING_TABLE_END_BLACK = KING_TABLE_END[::-1]


def eval(board: chess.Board, root_color):
    white_eval = 0
    black_eval = 0
    is_endgame = False
    # xd = []
    # kk = []
    board_hash = zobrist_hash(board)
    if board_hash in trans:
        white_eval, black_eval = trans[board_hash]
    else:
        if board.queens == 0:
            is_endgame = True
        if board.queens & board.occupied_co[chess.WHITE] == 0 and board.queens & board.occupied_co[chess.BLACK] != 0:
            is_endgame = count_ones((board.knights | board.bishops | board.rooks) & board.occupied_co[chess.WHITE]) <= 1
        elif board.queens & board.occupied_co[chess.BLACK] == 0 and board.queens & board.occupied_co[chess.WHITE] != 0:
            is_endgame = count_ones((board.knights | board.bishops | board.rooks) & board.occupied_co[chess.BLACK]) <= 1
        # xd = []
        # kk = []
        for figure, weight, table in zip(
                [board.white_pawns, board.white_knights, board.white_bishops, board.white_rooks, board.white_queens, board.white_kings],
                [PAWN, KNIGHT, BISHOP, ROOK, QUEEN, KING],
                [PAWN_TABLE, KNIGHT_TABLE, BISHOP_TABLE,
                 ROOK_TABLE, QUEEN_TABLE,
                 KING_TABLE if not is_endgame else KING_TABLE_END]):

            white_eval += count_ones(figure) * weight
            # xd.append(count_ones(figure) * weight)
            if figure:
                white_eval += sum(table[i] for i in range(64) if figure & (1 << i))
                # kk.append(sum(table[i] for i in range(64) if figure & (1 << i)))
        # pass
        # xd =[]
        # kk = []
        for figure, weight, table in zip(
                [board.black_pawns, board.black_knights, board.black_bishops, board.black_rooks, board.black_queens, board.black_kings],
                [PAWN, KNIGHT, BISHOP, ROOK, QUEEN, KING],
                [PAWN_TABLE_BLACK, KNIGHT_TABLE_BLACK, BISHOP_TABLE_BLACK,
                 ROOK_TABLE_BLACK, QUEEN_TABLE_BLACK,
                 KING_TABLE_BLACK if not is_endgame else KING_TABLE_END_BLACK]):

            black_eval += count_ones(figure) * weight
            # xd.append(count_ones(figure) * weight)
            if figure:
                black_eval += sum(table[i] for i in range(64) if figure & (1 << i))
                # kk.append(sum(table[i] for i in range(64) if figure & (1 << i)))

        trans[board_hash] = [white_eval, black_eval]

    return (white_eval - black_eval) if root_color else (black_eval - white_eval)


def count_ones(n):
    count = 0
    while n:
        n &= n - 1
        count += 1
    return count

def sum_values_where_bytes_are_1(byte_variable, value_array):
    byte_size = len(value_array)
    sum_values = 0
    for i in range(byte_size):
        if byte_variable & (1 << i):
            sum_values += value_array[i]
    return sum_values
def print_byte(byte):
    if type(byte) == list:
        for b in byte:
            string = format(b, '0{}b'.format(64))
            for i in range(0, len(string), 8):
                print(string[i:i + 8][::-1])
            print("\n")
    else:
        string = format(byte, '0{}b'.format(64))
        for i in range(0, len(string), 8):
            print(string[i:i + 8])
    print("\n")


def MCTS_self_play(num_games):
    for idxx in range(0, num_games):
        current_board = chess.Board("r1bqkbnr/pp1ppppp/2n5/2p5/2B1P3/5N2/PPPP1PPP/RNBQK2R b KQkq - 0 1")
        states = []
        while current_board.fullmove_number <= 200:
            draw_counter = 0
            for s in states:
                if np.array_equal(current_board.board_fen(), s):
                    draw_counter += 1
            if draw_counter == 3:
                break
            states.append(copy.deepcopy(current_board.board_fen()))
            t = time.time()
            best_move = UCT_search(current_board, 4000000)
            print("time: ", time.time() - t)
            current_board.push(best_move)
            global trans
            trans = {}
            print(current_board, eval(current_board, not current_board.turn), best_move, chess.Move(*best_move))
            # move = input("Enter move: ")
            # move = chess.Move.from_uci(move)
            # current_board.push((move.from_square, move.to_square, move.promotion, move.drop))

            if current_board.is_checkmate():
                if current_board.turn:  # black wins
                    print("Black wins")
                else:  # white wins
                    print("White wins")
                break


MCTS_self_play(50)
