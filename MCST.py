import pickle
import os
import collections

import chess
import numpy as np
import math
import encoder_decoder as ed
import copy
import torch
import torch.multiprocessing as mp
from alpha_net import ChessNet
import datetime
import time
from stockfish import Stockfish
import math

fish = Stockfish(r"C:\Users\Mateu\Desktop\stockfish.exe",depth=7, parameters={"Threads": 4, "Hash": 2048})
# https://github.com/geochri/AlphaZero_Chess


class UCTNode:
    def __init__(self, game, move, parent=None):
        self.game = game  # state s
        self.move = move  # action index
        self.is_expanded = False
        self.parent = parent
        self.children = {}
        self.child_priors = np.zeros([4672], dtype=np.float32)
        self.child_total_value = np.zeros([4672], dtype=np.float32)
        self.child_number_visits = np.zeros([4672], dtype=np.float32)
        self.action_idxes = []

    @property
    def number_visits(self):
        return self.parent.child_number_visits[self.move]

    @number_visits.setter
    def number_visits(self, value):
        self.parent.child_number_visits[self.move] = value

    @property
    def total_value(self):
        return self.parent.child_total_value[self.move]

    @total_value.setter
    def total_value(self, value):
        self.parent.child_total_value[self.move] = value

    def child_Q(self):
        return self.child_total_value / (1 + self.child_number_visits)

    def child_U(self):
        return math.sqrt(self.number_visits) * (
                abs(self.child_priors) / (1 + self.child_number_visits)
        )

    def best_child(self):
        if self.action_idxes != []:
            bestmove = self.child_Q() + self.child_U()
            return self.action_idxes[np.argmax(bestmove[self.action_idxes])]

        return np.argmax(self.child_Q() + self.child_U())

    def select_leaf(self):
        current = self
        while current.is_expanded:
            best_move = current.best_child()
            current = current.maybe_add_child(best_move)
        return current

    def add_dirichlet_noise(self, action_idxs, child_priors):
        valid_child_priors = child_priors[
            action_idxs
        ]  # select only legal moves entries in child_priors array
        valid_child_priors = 0.75 * valid_child_priors + 0.25 * np.random.dirichlet(
            np.zeros([len(valid_child_priors)], dtype=np.float32) + 0.3
        )
        child_priors[action_idxs] = valid_child_priors
        return child_priors

    def expand(self, child_priors):
        self.is_expanded = True
        action_idxs = []
        c_p = child_priors
        y = list(self.game.generate_legal_moves())
        y = [(x.from_square, x.to_square, x.promotion) for x in y]
        y = [
            [
                (from_square // 8, from_square % 8),
                (to_square // 8, to_square % 8),
                {5: None, 4: "rook", 3: "bishop", 2: "knight", None: None}[promotion],
            ]
            for (from_square, to_square, promotion) in y
        ]

        for action in y:
            if action != []:
                initial_pos, final_pos, underpromote = action
                action_idxs.append(
                    ed.encode_action(self.game, initial_pos, final_pos, underpromote)
                )
        if not action_idxs:
            self.is_expanded = False
        self.action_idxes = action_idxs
        for i in range(len(child_priors)):  # mask all illegal actions
            if i not in action_idxs:
                c_p[i] = 0.0000000000
        if self.parent.parent is None:  # add dirichlet noise to child_priors in root node TODO TODO tODO
            c_p = self.add_dirichlet_noise(action_idxs, c_p)
        self.child_priors = c_p

    def decode_n_move_pieces(self, board, move):
        i_pos, f_pos, prom = ed.decode_action(board, move)
        for i, f, p in zip(i_pos, f_pos, prom):
            letters = "abcdefgh"
            a, b = i
            c, d = f
            move_string = letters[b] + str(a + 1) + letters[d] + str(c + 1)
            if p is not None:
                move_string += p.lower()
            board.push_san(move_string)
        return board

    def maybe_add_child(self, move):
        if move not in self.children:
            copy_board = chess.Board(self.game.fen())
            copy_board = self.decode_n_move_pieces(copy_board, move)
            self.children[move] = UCTNode(copy_board, move, parent=self)
        return self.children[move]

    def backup(self, value_estimate: float):
        if value_estimate == 0:
            value_estimate = 0.0000000001

        self.total_value = value_estimate
        self.number_visits += 1
        print("chuj")
        current = self.parent
        while current.parent is not None:
            current.number_visits += 1
            if current.game.turn:
                current.total_value = np.max(self.child_total_value[self.child_total_value != 0])
            else:
                current.total_value = np.min(self.child_total_value[self.child_total_value != 0])
            print(current.game,current.total_value,current.child_total_value[current.child_total_value != 0])
            current = current.parent


class DummyNode(object):
    def __init__(self):
        self.parent = None
        self.child_total_value = collections.defaultdict(float)
        self.child_number_visits = collections.defaultdict(float)


def UCT_search(game_state, num_reads, net):
    root = UCTNode(game_state, move=None, parent=DummyNode())
    for i in range(num_reads):
        leaf = root.select_leaf()
        encoded_s = ed.encode_board(leaf.game)
        encoded_s = encoded_s.transpose(2, 0, 1)
        encoded_s = torch.from_numpy(encoded_s).float()  # .cuda()
        child_priors, value_estimate = net(encoded_s)
        child_priors = child_priors.detach().cpu().numpy().reshape(-1)
        value_estimate = value_estimate.item()
        fish.set_fen_position(leaf.game.fen())
        eval = fish.get_evaluation()
        if eval["type"] == "cp":
            value = eval["value"]
            if value > 0:
                eval = min(value / 5000, 1)
            elif value < 0:
                eval = max(value / 5000, -1)
            else:
                eval = 0
        elif eval["type"] == "mate":
            if eval["value"] > 0:
                eval = 1
            else:
                eval = -1
        value_estimate = eval
        if leaf.game.is_checkmate():
            leaf.backup(value_estimate)
            continue
        leaf.expand(child_priors)  # need to make sure valid moves
        leaf.backup(value_estimate)

    return np.argmax(root.child_number_visits), root


def do_decode_n_move_pieces(board, move):
    i_pos, f_pos, prom = ed.decode_action(board, move)
    for i, f, p in zip(i_pos, f_pos, prom):
        letters = "abcdefgh"
        a, b = i
        c, d = f
        move_string = letters[b] + str(a + 1) + letters[d] + str(c + 1)
        if p is not None:
            move_string += p.lower()
        board.push_san(move_string)
    return board


def get_policy(root):
    policy = np.zeros([4672], dtype=np.float32)
    for idx in np.where(root.child_number_visits != 0)[0]:
        policy[idx] = root.child_number_visits[idx] / root.child_number_visits.sum()
    return policy


def save_as_pickle(filename, data):
    completeName = os.path.join("datasets/iter2/", filename)
    with open(completeName, "wb") as output:
        pickle.dump(data, output)


def load_pickle(filename):
    completeName = os.path.join("datasets/", filename)
    with open(completeName, "rb") as pkl_file:
        data = pickle.load(pkl_file)
    return data


def MCTS_self_play(chessnet, num_games, cpu):
    for idxx in range(0, num_games):
        current_board = chess.Board("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/R1BQKBNR w KQkq - 0 1")
        dataset = []  # to get state, policy, value for neural network training
        states = []
        value = 0
        while current_board.fullmove_number <= 100:
            draw_counter = 0
            for s in states:
                if np.array_equal(current_board.board_fen(), s):
                    draw_counter += 1
            if draw_counter == 3:  # draw by repetition
                break
            states.append(copy.deepcopy(current_board.board_fen()))
            board_state = copy.deepcopy(ed.encode_board(current_board))
            t = time.time()
            best_move, root = UCT_search(current_board, 40, chessnet)
            print(root.child_total_value[root.child_total_value != 0])
            print("time: ", time.time() - t)
            current_board = do_decode_n_move_pieces(
                current_board, best_move
            )
            policy = get_policy(root)
            # dataset.append([board_state, policy])
            fish.set_fen_position(current_board.fen())
            eval = fish.get_evaluation()

            if eval["type"] == "cp":
                value = eval["value"]
                if value > 0:
                    eval = min(value/5000, 1)
                elif value < 0:
                    eval = max(value/5000, -1)
                else:
                    eval = 0
            elif eval["type"] == "mate":
                if eval["value"] > 0:
                    eval = 1
                else:
                    eval = -1
            print(eval, board_state[1,1,12], current_board.fen())
            dataset.append([board_state, policy, eval])
            print(current_board, current_board.fullmove_number)
            print(" ")
            if current_board.is_checkmate():
                if current_board.turn:  # black wins
                    value = -1
                    print("Black wins")
                else:  # white wins
                    value = 1
                    print("White wins")
                break
            else:
                value = 0

        # dataset_p = []
        # for idx, data in enumerate(dataset):
        #     s, p = data
        #     if idx == 0:
        #         dataset_p.append([s, p, 0])
        #     else:
        #         dataset_p.append([s, p, value])
        dataset_p = [x for x in dataset]
        del dataset
        print("dataset_cpu%i_%i_%s_%i" % (cpu, idxx, datetime.datetime.now().strftime("%Y-%m-%d_%H:%M"), value))
        save_as_pickle(
            "dataset_cpu%i_%i_%s_%i" % (cpu, idxx, datetime.datetime.now().strftime("%Y-%m-%d_%H-%M"), value),
            dataset_p,
        )


if __name__ == "__main__":
    net_to_play = "current_net_trained8_iter1.pth.tar"
    mp.set_start_method("spawn", force=True)
    net = ChessNet()
    cuda = torch.cuda.is_available()
    if cuda:
        net.cuda()
    net.share_memory()
    net.eval()
    print("hi")

    current_net_filename = os.path.join("./model_data/", net_to_play)
    # check if file exists
    if not os.path.isfile(current_net_filename):
        torch.save({"state_dict": net.state_dict()}, os.path.join("./model_data/", net_to_play))
        print("created model")

    checkpoint = torch.load(current_net_filename)
    net.load_state_dict(checkpoint['state_dict'])

    # processes = []
    # for i in range(1):
    #     p = mp.Process(target=MCTS_self_play, args=(net, 50, i))
    #     p.start()
    #     processes.append(p)
    # for p in processes:
    #     p.join()

    MCTS_self_play(net, 50, 0)
