import chess
import numpy as np


def encode_board(board):
    encoded = np.zeros([64, 19]).astype(int)
    encoder_dict = {
        "R": 0,
        "N": 1,
        "B": 2,
        "Q": 3,
        "K": 4,
        "P": 5,
        "r": 6,
        "n": 7,
        "b": 8,
        "q": 9,
        "k": 10,
        "p": 11,
    }
    for square in chess.SQUARES_180:
        piece = board.piece_at(square)

        if piece:
            encoded[63-square, encoder_dict[piece.symbol()]] = 1

    encoded = encoded.reshape(8, 8, 19)

    encoded[:, :, 12] = int(board.turn)
    encoded[:, :, 13] = board.has_queenside_castling_rights(chess.WHITE)
    encoded[:, :, 14] = board.has_kingside_castling_rights(chess.WHITE)
    encoded[:, :, 15] = board.has_queenside_castling_rights(chess.BLACK)
    encoded[:, :, 16] = board.has_kingside_castling_rights(chess.BLACK)
    encoded[:, :, 17] = board.fullmove_number

    # print("\n".join(str(board).split("\n")[::-1]))

    if board.ep_square:
        square = 63-board.ep_square
        encoded[square//8,square%8,18] = 1

    return encoded


def encode_action(board, initial_pos, final_pos, underpromote=None):
    encoded = np.zeros([8, 8, 73]).astype(int)
    i, j = initial_pos
    x, y = final_pos
    dx, dy = x - i, y - j
    piece = board.piece_at(chess.Square(i * 8 + j)).symbol()

    if piece in ["R", "B", "Q", "K", "P", "r", "b", "q", "k", "p"] and underpromote in [
        None,
        "queen",
    ]:  # queen-like moves
        if dx != 0 and dy == 0:  # north-south idx 0-13
            if dx < 0:
                idx = 7 + dx
            elif dx > 0:
                idx = 6 + dx
        if dx == 0 and dy != 0:  # east-west idx 14-27
            if dy < 0:
                idx = 21 + dy
            elif dy > 0:
                idx = 20 + dy
        if dx == dy:  # NW-SE idx 28-41
            if dx < 0:
                idx = 35 + dx
            if dx > 0:
                idx = 34 + dx
        if dx == -dy:  # NE-SW idx 42-55
            if dx < 0:
                idx = 49 + dx
            if dx > 0:
                idx = 48 + dx
    if piece in ["n", "N"]:  # Knight moves 56-63
        if (x, y) == (i + 2, j - 1):
            idx = 56
        elif (x, y) == (i + 2, j + 1):
            idx = 57
        elif (x, y) == (i + 1, j - 2):
            idx = 58
        elif (x, y) == (i - 1, j - 2):
            idx = 59
        elif (x, y) == (i - 2, j + 1):
            idx = 60
        elif (x, y) == (i - 2, j - 1):
            idx = 61
        elif (x, y) == (i - 1, j + 2):
            idx = 62
        elif (x, y) == (i + 1, j + 2):
            idx = 63
    if piece in ["p", "P"] and (x == 0 or x == 7) and underpromote is not None:  # underpromotions
        if abs(dx) == 1 and dy == 0:
            if underpromote == "rook":
                idx = 64
            if underpromote == "knight":
                idx = 65
            if underpromote == "bishop":
                idx = 66
        if abs(dx) == 1 and dy == -1:
            if underpromote == "rook":
                idx = 67
            if underpromote == "knight":
                idx = 68
            if underpromote == "bishop":
                idx = 69
        if abs(dx) == 1 and dy == 1:
            if underpromote == "rook":
                idx = 70
            if underpromote == "knight":
                idx = 71
            if underpromote == "bishop":
                idx = 72
    encoded[i, j, idx] = 1
    encoded = encoded.reshape(-1)
    encoded = np.where(encoded == 1)[0][0]  # index of action
    return encoded


def decode_action(board, encoded):
    encoded_a = np.zeros([4672])
    encoded_a[encoded] = 1
    encoded_a = encoded_a.reshape(8, 8, 73)
    a, b, c = np.where(encoded_a == 1)  # i,j,k = i[0],j[0],k[0]
    i_pos, f_pos, prom = [], [], []
    for pos in zip(a, b, c):
        i, j, k = pos

        initial_pos = (i, j)
        promoted = None
        if 0 <= k <= 13:
            dy = 0
            if k < 7:
                dx = k - 7
            else:
                dx = k - 6
            final_pos = (i + dx, j + dy)
        elif 14 <= k <= 27:
            dx = 0
            if k < 21:
                dy = k - 21
            else:
                dy = k - 20
            final_pos = (i + dx, j + dy)
        elif 28 <= k <= 41:
            if k < 35:
                dy = k - 35
            else:
                dy = k - 34
            dx = dy
            final_pos = (i + dx, j + dy)
        elif 42 <= k <= 55:
            if k < 49:
                dx = k - 49
            else:
                dx = k - 48
            dy = -dx
            final_pos = (i + dx, j + dy)
        elif 56 <= k <= 63:
            if k == 56:
                final_pos = (i + 2, j - 1)
            elif k == 57:
                final_pos = (i + 2, j + 1)
            elif k == 58:
                final_pos = (i + 1, j - 2)
            elif k == 59:
                final_pos = (i - 1, j - 2)
            elif k == 60:
                final_pos = (i - 2, j + 1)
            elif k == 61:
                final_pos = (i - 2, j - 1)
            elif k == 62:
                final_pos = (i - 1, j + 2)
            elif k == 63:
                final_pos = (i + 1, j + 2)
        else:
            if k == 64:
                if board.turn == 0:
                    final_pos = (i - 1, j)
                    promoted = "R"
                else:
                    final_pos = (i + 1, j)
                    promoted = "r"
            if k == 65:
                if board.turn == 0:
                    final_pos = (i - 1, j)
                    promoted = "N"
                else:
                    final_pos = (i + 1, j)
                    promoted = "n"
            if k == 66:
                if board.turn == 0:
                    final_pos = (i - 1, j)
                    promoted = "B"
                else:
                    final_pos = (i + 1, j)
                    promoted = "b"
            if k == 67:
                if board.turn == 0:
                    final_pos = (i - 1, j - 1)
                    promoted = "R"
                else:
                    final_pos = (i + 1, j - 1)
                    promoted = "r"
            if k == 68:
                if board.turn == 0:
                    final_pos = (i - 1, j - 1)
                    promoted = "N"
                else:
                    final_pos = (i + 1, j - 1)
                    promoted = "n"
            if k == 69:
                if board.turn == 0:
                    final_pos = (i - 1, j - 1)
                    promoted = "B"
                else:
                    final_pos = (i + 1, j - 1)
                    promoted = "b"
            if k == 70:
                if board.turn == 0:
                    final_pos = (i - 1, j + 1)
                    promoted = "R"
                else:
                    final_pos = (i + 1, j + 1)
                    promoted = "r"
            if k == 71:
                if board.turn == 0:
                    final_pos = (i - 1, j + 1)
                    promoted = "N"
                else:
                    final_pos = (i + 1, j + 1)
                    promoted = "n"
            if k == 72:
                if board.turn == 0:
                    final_pos = (i - 1, j + 1)
                    promoted = "B"
                else:
                    final_pos = (i + 1, j + 1)
                    promoted = "b"
        if (
            board.piece_at(chess.Square(i * 8 + j)).symbol() in ["P", "p"]
            and final_pos[0] in [0, 7]
            and promoted is None
        ):  # auto-queen promotion for pawn
            if board.turn == 0:
                promoted = "Q"
            else:
                promoted = "q"

        i_pos.append(initial_pos)
        f_pos.append(final_pos), prom.append(promoted)
    return i_pos, f_pos, prom