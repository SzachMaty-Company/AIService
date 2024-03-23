# This file is part of the python-chess library.
# Copyright (C) 2012-2021 Niklas Fiekas <niklas.fiekas@backscattering.de>
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program. If not, see <http://www.gnu.org/licenses/>.

"""
A chess library with move generation and validation,
Polyglot opening book probing, PGN reading and writing,
Gaviota tablebase probing,
Syzygy tablebase probing, and UCI engine communication.
"""

from __future__ import annotations

__author__ = "Niklas Fiekas"

__email__ = "niklas.fiekas@backscattering.de"

__version__ = "1.10.0"

import dataclasses
import math
import re
import typing

from typing import (
    ClassVar,
    Callable,
    Dict,
    Iterable,
    Iterator,
    List,
    Mapping,
    Optional,
    SupportsInt,
    Tuple,
    TypeVar,
    Union,
)

try:
    from typing import Literal

    _EnPassantSpec = Literal["legal", "fen", "xfen"]
except ImportError:
    # Before Python 3.8.
    _EnPassantSpec = str  # type: ignore
from chess import SquareSet, LegalMoveGenerator, Piece

Color = bool
COLORS = [WHITE, BLACK] = [True, False]
COLOR_NAMES = ["black", "white"]

PieceType = int
PIECE_TYPES = [PAWN, KNIGHT, BISHOP, ROOK, QUEEN, KING] = range(1, 7)
PIECE_SYMBOLS = [None, "p", "n", "b", "r", "q", "k"]
NEW_PIECE_SYMBOLS = [None, "p", "n", "b", "r", "q", "k", "P", "N", "B", "R", "Q", "K"]
PIECE_NAMES = [None, "pawn", "knight", "bishop", "rook", "queen", "king"]

FROM_SQUARE = 0
TO_SQUARE = 1


def piece_symbol(piece_type: PieceType) -> str:
    return typing.cast(str, PIECE_SYMBOLS[piece_type])


def piece_name(piece_type: PieceType) -> str:
    return typing.cast(str, PIECE_NAMES[piece_type])


UNICODE_PIECE_SYMBOLS = {
    "R": "♖",
    "r": "♜",
    "N": "♘",
    "n": "♞",
    "B": "♗",
    "b": "♝",
    "Q": "♕",
    "q": "♛",
    "K": "♔",
    "k": "♚",
    "P": "♙",
    "p": "♟",
}

FILE_NAMES = ["a", "b", "c", "d", "e", "f", "g", "h"]

RANK_NAMES = ["1", "2", "3", "4", "5", "6", "7", "8"]

STARTING_FEN = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
"""The FEN for the standard chess starting position."""

STARTING_BOARD_FEN = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR"
"""The board part of the FEN for the standard chess starting position."""


class InvalidMoveError(ValueError):
    """Raised when move notation is not syntactically valid"""


Square = int
SQUARES = [
    A1,
    B1,
    C1,
    D1,
    E1,
    F1,
    G1,
    H1,
    A2,
    B2,
    C2,
    D2,
    E2,
    F2,
    G2,
    H2,
    A3,
    B3,
    C3,
    D3,
    E3,
    F3,
    G3,
    H3,
    A4,
    B4,
    C4,
    D4,
    E4,
    F4,
    G4,
    H4,
    A5,
    B5,
    C5,
    D5,
    E5,
    F5,
    G5,
    H5,
    A6,
    B6,
    C6,
    D6,
    E6,
    F6,
    G6,
    H6,
    A7,
    B7,
    C7,
    D7,
    E7,
    F7,
    G7,
    H7,
    A8,
    B8,
    C8,
    D8,
    E8,
    F8,
    G8,
    H8,
] = range(64)

SQUARE_NAMES = [f + r for r in RANK_NAMES for f in FILE_NAMES]


def square(file_index: int, rank_index: int) -> Square:
    """Gets a square number by file and rank index."""
    return rank_index * 8 + file_index


def square_file(square: Square) -> int:
    """Gets the file index of the square where ``0`` is the a-file."""
    return square & 7


def square_rank(square: Square) -> int:
    """Gets the rank index of the square where ``0`` is the first rank."""
    return square >> 3


def square_distance(a: Square, b: Square) -> int:
    """
    Gets the Chebyshev distance (i.e., the number of king steps) from square *a* to *b*.
    """
    return max(abs(square_file(a) - square_file(b)), abs(square_rank(a) - square_rank(b)))


def square_mirror(square: Square) -> Square:
    """Mirrors the square vertically."""
    return square ^ 0x38


SQUARES_180 = [square_mirror(sq) for sq in SQUARES]


Bitboard = int
BB_EMPTY = 0
BB_ALL = 0xFFFF_FFFF_FFFF_FFFF

BB_SQUARES = [
    BB_A1,
    BB_B1,
    BB_C1,
    BB_D1,
    BB_E1,
    BB_F1,
    BB_G1,
    BB_H1,
    BB_A2,
    BB_B2,
    BB_C2,
    BB_D2,
    BB_E2,
    BB_F2,
    BB_G2,
    BB_H2,
    BB_A3,
    BB_B3,
    BB_C3,
    BB_D3,
    BB_E3,
    BB_F3,
    BB_G3,
    BB_H3,
    BB_A4,
    BB_B4,
    BB_C4,
    BB_D4,
    BB_E4,
    BB_F4,
    BB_G4,
    BB_H4,
    BB_A5,
    BB_B5,
    BB_C5,
    BB_D5,
    BB_E5,
    BB_F5,
    BB_G5,
    BB_H5,
    BB_A6,
    BB_B6,
    BB_C6,
    BB_D6,
    BB_E6,
    BB_F6,
    BB_G6,
    BB_H6,
    BB_A7,
    BB_B7,
    BB_C7,
    BB_D7,
    BB_E7,
    BB_F7,
    BB_G7,
    BB_H7,
    BB_A8,
    BB_B8,
    BB_C8,
    BB_D8,
    BB_E8,
    BB_F8,
    BB_G8,
    BB_H8,
] = [1 << sq for sq in SQUARES]

BB_CORNERS = BB_A1 | BB_H1 | BB_A8 | BB_H8
BB_CENTER = BB_D4 | BB_E4 | BB_D5 | BB_E5

BB_LIGHT_SQUARES = 0x55AA_55AA_55AA_55AA
BB_DARK_SQUARES = 0xAA55_AA55_AA55_AA55

BB_FILES = [
    BB_FILE_A,
    BB_FILE_B,
    BB_FILE_C,
    BB_FILE_D,
    BB_FILE_E,
    BB_FILE_F,
    BB_FILE_G,
    BB_FILE_H,
] = [0x0101_0101_0101_0101 << i for i in range(8)]

BB_RANKS = [
    BB_RANK_1,
    BB_RANK_2,
    BB_RANK_3,
    BB_RANK_4,
    BB_RANK_5,
    BB_RANK_6,
    BB_RANK_7,
    BB_RANK_8,
] = [0xFF << (8 * i) for i in range(8)]

BB_BACKRANKS = BB_RANK_1 | BB_RANK_8

class ChessPiece:
    piece_type: PieceType
    color: Color
    position: int


def lsb(bb: Bitboard) -> int:
    return (bb & -bb).bit_length() - 1


def msb(bb: Bitboard) -> int:
    return bb.bit_length() - 1


def scan_reversed(bb: Bitboard) -> Iterator[Square]:
    while bb:
        r = bb.bit_length() - 1
        yield r
        bb ^= BB_SQUARES[r]


# Python 3.10 or fallback.
popcount: Callable[[Bitboard], int] = getattr(int, "bit_count", lambda bb: bin(bb).count("1"))


def _sliding_attacks(square: Square, occupied: Bitboard, deltas: Iterable[int]) -> Bitboard:
    attacks = BB_EMPTY

    for delta in deltas:
        sq = square

        while True:
            sq += delta
            if not (0 <= sq < 64) or square_distance(sq, sq - delta) > 2:
                break

            attacks |= BB_SQUARES[sq]

            if occupied & BB_SQUARES[sq]:
                break

    return attacks


def _step_attacks(square: Square, deltas: Iterable[int]) -> Bitboard:
    return _sliding_attacks(square, BB_ALL, deltas)


BB_KNIGHT_ATTACKS = [_step_attacks(sq, [17, 15, 10, 6, -17, -15, -10, -6]) for sq in SQUARES]
BB_KING_ATTACKS = [_step_attacks(sq, [9, 8, 7, 1, -9, -8, -7, -1]) for sq in SQUARES]
BB_PAWN_ATTACKS = [[_step_attacks(sq, deltas) for sq in SQUARES] for deltas in [[-7, -9], [7, 9]]]


def _edges(square: Square) -> Bitboard:
    return ((BB_RANK_1 | BB_RANK_8) & ~BB_RANKS[square_rank(square)]) | (
        (BB_FILE_A | BB_FILE_H) & ~BB_FILES[square_file(square)]
    )


def _carry_rippler(mask: Bitboard) -> Iterator[Bitboard]:
    # Carry-Rippler trick to iterate subsets of mask.
    subset = BB_EMPTY
    while True:
        yield subset
        subset = (subset - mask) & mask
        if not subset:
            break


def _attack_table(deltas: List[int]) -> Tuple[List[Bitboard], List[Dict[Bitboard, Bitboard]]]:
    mask_table = []
    attack_table = []

    for square in SQUARES:
        attacks = {}

        mask = _sliding_attacks(square, 0, deltas) & ~_edges(square)
        for subset in _carry_rippler(mask):
            attacks[subset] = _sliding_attacks(square, subset, deltas)

        attack_table.append(attacks)
        mask_table.append(mask)

    return mask_table, attack_table


BB_DIAG_MASKS, BB_DIAG_ATTACKS = _attack_table([-9, -7, 7, 9])
BB_FILE_MASKS, BB_FILE_ATTACKS = _attack_table([-8, 8])
BB_RANK_MASKS, BB_RANK_ATTACKS = _attack_table([-1, 1])


def _rays() -> List[List[Bitboard]]:
    rays = []
    for a, bb_a in enumerate(BB_SQUARES):
        rays_row = []
        for b, bb_b in enumerate(BB_SQUARES):
            if BB_DIAG_ATTACKS[a][0] & bb_b:
                rays_row.append((BB_DIAG_ATTACKS[a][0] & BB_DIAG_ATTACKS[b][0]) | bb_a | bb_b)
            elif BB_RANK_ATTACKS[a][0] & bb_b:
                rays_row.append(BB_RANK_ATTACKS[a][0] | bb_a)
            elif BB_FILE_ATTACKS[a][0] & bb_b:
                rays_row.append(BB_FILE_ATTACKS[a][0] | bb_a)
            else:
                rays_row.append(BB_EMPTY)
        rays.append(rays_row)
    return rays


BB_RAYS = _rays()


def ray(a: Square, b: Square) -> Bitboard:
    return BB_RAYS[a][b]


def between(a: Square, b: Square) -> Bitboard:
    bb = BB_RAYS[a][b] & ((BB_ALL << a) ^ (BB_ALL << b))
    return bb & (bb - 1)


SAN_REGEX = re.compile(r"^([NBKRQ])?([a-h])?([1-8])?[\-x]?([a-h][1-8])(=?[nbrqkNBRQK])?[\+#]?\Z")

FEN_CASTLING_REGEX = re.compile(r"^(?:-|[KQABCDEFGH]{0,2}[kqabcdefgh]{0,2})\Z")


@dataclasses.dataclass(unsafe_hash=True)
class Move:
    """
    Represents a move from a square to a square and possibly the promotion
    piece type.

    Drops and null moves are supported.
    """

    from_square: Square
    """The source square."""

    to_square: Square
    """The target square."""

    promotion: Optional[PieceType] = None
    """The promotion piece type or ``None``."""

    drop: Optional[PieceType] = None
    """The drop piece type or ``None``."""

    piece: int|None = None

    moving_piece: int|None = None

    eval: int|None = None

    def uci(self) -> str:
        """
        Gets a UCI string for the move.

        For example, a move from a7 to a8 would be ``a7a8`` or ``a7a8q``
        (if the latter is a promotion to a queen).

        The UCI representation of a null move is ``0000``.
        """
        if self.drop:
            return piece_symbol(self.drop).upper() + "@" + SQUARE_NAMES[self.to_square]
        elif self.promotion:
            return SQUARE_NAMES[self.from_square] + SQUARE_NAMES[self.to_square] + piece_symbol(self.promotion)
        elif self:
            return SQUARE_NAMES[self.from_square] + SQUARE_NAMES[self.to_square]
        else:
            return "0000"

    def __bool__(self) -> bool:
        return bool(self.from_square or self.to_square or self.promotion or self.drop)

    def __repr__(self) -> str:
        return f"Move.from_uci({self.uci()!r})"

    def __str__(self) -> str:
        return self.uci()

    @classmethod
    def from_uci(cls, uci: str) -> Move:
        """
        Parses a UCI string.

        :raises: :exc:`InvalidMoveError` if the UCI string is invalid.
        """
        if uci == "0000":
            return cls.null()
        elif len(uci) == 4 and "@" == uci[1]:
            try:
                drop = PIECE_SYMBOLS.index(uci[0].lower())
                square = SQUARE_NAMES.index(uci[2:])
            except ValueError:
                raise InvalidMoveError(f"invalid uci: {uci!r}")
            return cls(square, square, drop=drop)
        elif 4 <= len(uci) <= 5:
            try:
                from_square = SQUARE_NAMES.index(uci[0:2])
                to_square = SQUARE_NAMES.index(uci[2:4])
                promotion = PIECE_SYMBOLS.index(uci[4]) if len(uci) == 5 else None
            except ValueError:
                raise InvalidMoveError(f"invalid uci: {uci!r}")
            if from_square == to_square:
                raise InvalidMoveError(f"invalid uci (use 0000 for null moves): {uci!r}")
            return cls(from_square, to_square, promotion=promotion)
        else:
            raise InvalidMoveError(f"expected uci string to be of length 4 or 5: {uci!r}")

    @classmethod
    def null(cls) -> Move:
        """
        Gets a null move.

        A null move just passes the turn to the other side (and possibly
        forfeits en passant capturing). Null moves evaluate to ``False`` in
        boolean contexts.

        >>> import chess
        >>>
        >>> bool(chess.Move.null())
        False
        """
        return cls(0, 0)


BaseBoardT = TypeVar("BaseBoardT", bound="BaseBoard")


class BaseBoard:
    """
    A board representing the position of chess pieces. See
    :class:`~chess.Board` for a full board with move generation.

    The board is initialized with the standard chess starting position, unless
    otherwise specified in the optional *board_fen* argument. If *board_fen*
    is ``None``, an empty board is created.
    """

    def __init__(self, board_fen: Optional[str] = STARTING_BOARD_FEN) -> None:
        self.occupied_co = [BB_EMPTY, BB_EMPTY]

        if board_fen is None:
            self._clear_board()
        else:
            self._set_board_fen(board_fen)

    def _clear_board(self) -> None:
        self.pawns = BB_EMPTY
        self.white_pawns = BB_EMPTY
        self.black_pawns = BB_EMPTY
        self.knights = BB_EMPTY
        self.white_knights = BB_EMPTY
        self.black_knights = BB_EMPTY
        self.bishops = BB_EMPTY
        self.white_bishops = BB_EMPTY
        self.black_bishops = BB_EMPTY
        self.rooks = BB_EMPTY
        self.white_rooks = BB_EMPTY
        self.black_rooks = BB_EMPTY
        self.queens = BB_EMPTY
        self.white_queens = BB_EMPTY
        self.black_queens = BB_EMPTY
        self.kings = BB_EMPTY
        self.white_kings = BB_EMPTY
        self.black_kings = BB_EMPTY

        self.promoted = BB_EMPTY

        self.occupied_co[WHITE] = BB_EMPTY
        self.occupied_co[BLACK] = BB_EMPTY
        self.occupied = BB_EMPTY

    def pieces_mask(self, piece_type: PieceType, color: Color) -> Bitboard:
        if piece_type == PAWN:
            if color == WHITE:
                return self.white_pawns
            else:
                return self.black_pawns
        elif piece_type == KNIGHT:
            if color == WHITE:
                return self.white_knights
            else:
                return self.black_knights
        elif piece_type == BISHOP:
            if color == WHITE:
                return self.white_bishops
            else:
                return self.black_bishops
        elif piece_type == ROOK:
            if color == WHITE:
                return self.white_rooks
            else:
                return self.black_rooks
        elif piece_type == QUEEN:
            if color == WHITE:
                return self.white_queens
            else:
                return self.black_queens
        elif piece_type == KING:
            if color == WHITE:
                return self.white_kings
            else:
                return self.black_kings
        else:
            assert False, f"expected PieceType, got {piece_type!r}"

    def piece_at(self, square: Square) -> Optional[Piece]:
        """Gets the :class:`piece <chess.Piece>` at the given square."""
        piece_type = self.piece_type_at(square)
        if piece_type:
            mask = BB_SQUARES[square]
            color = bool(self.occupied_co[WHITE] & mask)
            return Piece(piece_type, color)
        else:
            return None

    def piece_type_at(self, square: Square) -> Optional[PieceType]:
        """Gets the piece type at the given square."""
        mask = BB_SQUARES[square]

        if not self.occupied & mask:
            return None  # Early return
        elif self.pawns & mask:
            return PAWN
        elif self.knights & mask:
            return KNIGHT
        elif self.bishops & mask:
            return BISHOP
        elif self.rooks & mask:
            return ROOK
        elif self.queens & mask:
            return QUEEN
        else:
            return KING

    def king(self, color: Color) -> Optional[Square]:
        """
        Finds the king square of the given side. Returns ``None`` if there
        is no king of that color.

        In variants with king promotions, only non-promoted kings are
        considered.
        """
        king_mask = self.occupied_co[color] & self.kings & ~self.promoted
        return msb(king_mask) if king_mask else None

    def attacks_mask(self, square: Square) -> tuple[Bitboard, int]:
        bb_square = BB_SQUARES[square]

        if bb_square & self.pawns:
            color = bool(bb_square & self.occupied_co[WHITE])
            return (BB_PAWN_ATTACKS[color][square], PAWN)
        elif bb_square & self.knights:
            return (BB_KNIGHT_ATTACKS[square], KNIGHT)
        elif bb_square & self.kings:
            return (BB_KING_ATTACKS[square], KING)
        else:
            if bb_square & self.bishops:
                return (BB_DIAG_ATTACKS[square][BB_DIAG_MASKS[square] & self.occupied], BISHOP)
            if bb_square & self.queens:
                attacks = BB_DIAG_ATTACKS[square][BB_DIAG_MASKS[square] & self.occupied]
                attacks |= (
                    BB_RANK_ATTACKS[square][BB_RANK_MASKS[square] & self.occupied]
                    | BB_FILE_ATTACKS[square][BB_FILE_MASKS[square] & self.occupied]
                )
                return (attacks, QUEEN)
            if bb_square & self.rooks:
                return (
                    BB_RANK_ATTACKS[square][BB_RANK_MASKS[square] & self.occupied]
                    | BB_FILE_ATTACKS[square][BB_FILE_MASKS[square] & self.occupied], ROOK
                )


    def _attackers_mask(self, color: Color, square: Square, occupied: Bitboard) -> Bitboard:
        rank_pieces = BB_RANK_MASKS[square] & occupied
        file_pieces = BB_FILE_MASKS[square] & occupied
        diag_pieces = BB_DIAG_MASKS[square] & occupied

        queens_and_rooks = self.queens | self.rooks
        queens_and_bishops = self.queens | self.bishops

        attackers = (
            (BB_KING_ATTACKS[square] & self.kings)
            | (BB_KNIGHT_ATTACKS[square] & self.knights)
            | (BB_RANK_ATTACKS[square][rank_pieces] & queens_and_rooks)
            | (BB_FILE_ATTACKS[square][file_pieces] & queens_and_rooks)
            | (BB_DIAG_ATTACKS[square][diag_pieces] & queens_and_bishops)
            | (BB_PAWN_ATTACKS[not color][square] & self.pawns)
        )

        return attackers & self.occupied_co[color]

    def attackers_mask(self, color: Color, square: Square) -> Bitboard:
        return self._attackers_mask(color, square, self.occupied)

    def is_attacked_by(self, color: Color, square: Square) -> bool:
        """
        Checks if the given side attacks the given square.

        Pinned pieces still count as attackers. Pawns that can be captured
        en passant are **not** considered attacked.
        """
        return bool(self.attackers_mask(color, square))

    def pin_mask(self, color: Color, square: Square) -> Bitboard:
        king = self.king(color)
        if king is None:
            return BB_ALL

        square_mask = BB_SQUARES[square]

        for attacks, sliders in [
            (BB_FILE_ATTACKS, self.rooks | self.queens),
            (BB_RANK_ATTACKS, self.rooks | self.queens),
            (BB_DIAG_ATTACKS, self.bishops | self.queens),
        ]:
            rays = attacks[king][0]
            if rays & square_mask:
                snipers = rays & sliders & self.occupied_co[not color]
                for sniper in scan_reversed(snipers):
                    if between(sniper, king) & (self.occupied | square_mask) == square_mask:
                        return ray(king, sniper)

                break

        return BB_ALL

    def pin(self, color: Color, square: Square) -> SquareSet:
        """
        Detects an absolute pin (and its direction) of the given square to
        the king of the given color.

        >>> import chess
        >>>
        >>> board = chess.Board("rnb1k2r/ppp2ppp/5n2/3q4/1b1P4/2N5/PP3PPP/R1BQKBNR w KQkq - 3 7")
        >>> board.is_pinned(chess.WHITE, chess.C3)
        True
        >>> direction = board.pin(chess.WHITE, chess.C3)
        >>> direction
        SquareSet(0x0000_0001_0204_0810)
        >>> print(direction)
        . . . . . . . .
        . . . . . . . .
        . . . . . . . .
        1 . . . . . . .
        . 1 . . . . . .
        . . 1 . . . . .
        . . . 1 . . . .
        . . . . 1 . . .

        Returns a :class:`set of squares <chess.SquareSet>` that mask the rank,
        file or diagonal of the pin. If there is no pin, then a mask of the
        entire board is returned.
        """
        return SquareSet(self.pin_mask(color, square))

    def is_pinned(self, color: Color, square: Square) -> bool:
        """
        Detects if the given square is pinned to the king of the given color.
        """
        return self.pin_mask(color, square) != BB_ALL

    def _remove_piece_at(self, square: Square, piece_type: int | None = None, print=False) -> Optional[PieceType]:
        piece_type = piece_type if piece_type is not None else self.piece_type_at(square)
        mask = BB_SQUARES[square]
        mask2 = ~mask

        if print:
            self.print_byte(mask)
            self.print_byte(self.white_knights)
            self.print_byte(self.white_knights & mask2)
        if piece_type == PAWN:
            self.pawns ^= mask
            self.white_pawns &= mask2
            self.black_pawns &= mask2
        elif piece_type == KNIGHT:
            self.knights ^= mask
            self.white_knights &= mask2
            self.black_knights &=  mask2
        elif piece_type == BISHOP:
            self.bishops ^= mask
            self.white_bishops &=  mask2
            self.black_bishops &=  mask2
        elif piece_type == ROOK:
            self.rooks ^= mask
            self.white_rooks &=  mask2
            self.black_rooks &=  mask2
        elif piece_type == QUEEN:
            self.queens ^= mask
            self.white_queens &=  mask2
            self.black_queens &=  mask2
        elif piece_type == KING:
            self.kings ^= mask
            self.white_kings &= mask2
            self.black_kings &= mask2
        else:
            return None

        self.occupied ^= mask
        self.occupied_co[WHITE] &= ~mask
        self.occupied_co[BLACK] &= ~mask

        self.promoted &= ~mask

        return piece_type

    def print_byte(self, byte):
        if type(byte) == list:
            for b in byte:
                string = format(b, '0{}b'.format(64))
                for i in range(0, len(string), 8):
                    print(string[i:i + 8][::-1])
                print("\n")
        else:
            string = format(byte, '0{}b'.format(64))
            for i in range(0, len(string), 8):
                print(string[i:i + 8][::-1])
        print("\n")

    def _set_piece_at(self, square: Square, piece_type: PieceType, color: Color, promoted: bool = False, removed_piece_type: int|None=None) -> None:
        self._remove_piece_at(square, piece_type=removed_piece_type)
        # self.print_byte(self.white_kings)
        mask = BB_SQUARES[square]

        if piece_type == PAWN:
            self.pawns |= mask
            if color:
                self.white_pawns |= mask
            else:
                self.black_pawns |= mask
        elif piece_type == KNIGHT:
            self.knights |= mask
            if color:
                self.white_knights |= mask
            else:
                self.black_knights |= mask
        elif piece_type == BISHOP:
            self.bishops |= mask
            if color:
                self.white_bishops |= mask
            else:
                self.black_bishops |= mask
        elif piece_type == ROOK:
            self.rooks |= mask
            if color:
                self.white_rooks |= mask
            else:
                self.black_rooks |= mask
        elif piece_type == QUEEN:
            self.queens |= mask
            if color:
                self.white_queens |= mask
            else:
                self.black_queens |= mask
        elif piece_type == KING:
            self.kings |= mask
            if color:
                self.white_kings |= mask
            else:
                self.black_kings |= mask
        else:
            return

        self.occupied ^= mask
        self.occupied_co[color] ^= mask

        if promoted:
            self.promoted ^= mask

    def board_fen(self, *, promoted: Optional[bool] = False) -> str:
        """
        Gets the board FEN (e.g.,
        ``rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR``).
        """
        builder = []
        empty = 0

        for square in SQUARES_180:
            piece = self.piece_at(square)

            if not piece:
                empty += 1
            else:
                if empty:
                    builder.append(str(empty))
                    empty = 0
                builder.append(piece.symbol())
                if promoted and BB_SQUARES[square] & self.promoted:
                    builder.append("~")

            if BB_SQUARES[square] & BB_FILE_H:
                if empty:
                    builder.append(str(empty))
                    empty = 0

                if square != H1:
                    builder.append("/")

        return "".join(builder)

    def _set_board_fen(self, fen: str) -> None:
        # Compatibility with set_fen().
        fen = fen.strip()
        if " " in fen:
            raise ValueError(f"expected position part of fen, got multiple parts: {fen!r}")

        # Ensure the FEN is valid.
        rows = fen.split("/")
        if len(rows) != 8:
            raise ValueError(f"expected 8 rows in position part of fen: {fen!r}")

        # Validate each row.
        for row in rows:
            field_sum = 0
            previous_was_digit = False
            previous_was_piece = False

            for c in row:
                if c in ["1", "2", "3", "4", "5", "6", "7", "8"]:
                    if previous_was_digit:
                        raise ValueError(f"two subsequent digits in position part of fen: {fen!r}")
                    field_sum += int(c)
                    previous_was_digit = True
                    previous_was_piece = False
                elif c == "~":
                    if not previous_was_piece:
                        raise ValueError(f"'~' not after piece in position part of fen: {fen!r}")
                    previous_was_digit = False
                    previous_was_piece = False
                elif c.lower() in PIECE_SYMBOLS:
                    field_sum += 1
                    previous_was_digit = False
                    previous_was_piece = True
                else:
                    raise ValueError(f"invalid character in position part of fen: {fen!r}")

            if field_sum != 8:
                raise ValueError(f"expected 8 columns per row in position part of fen: {fen!r}")

        # Clear the board.
        self._clear_board()

        # Put pieces on the board.
        square_index = 0
        for c in fen:
            if c in ["1", "2", "3", "4", "5", "6", "7", "8"]:
                square_index += int(c)
            elif c.lower() in PIECE_SYMBOLS:
                piece = Piece.from_symbol(c)
                self._set_piece_at(SQUARES_180[square_index], piece.piece_type, piece.color)
                square_index += 1
            elif c == "~":
                self.promoted |= BB_SQUARES[SQUARES_180[square_index - 1]]

    def __repr__(self) -> str:
        return f"{type(self).__name__}({self.board_fen()!r})"

    def __str__(self) -> str:
        builder = []

        for square in SQUARES_180:
            piece = self.piece_at(square)

            if piece:
                builder.append(piece.symbol())
            else:
                builder.append(".")

            if BB_SQUARES[square] & BB_FILE_H:
                if square != H1:
                    builder.append("\n")
            else:
                builder.append(" ")

        return "".join(builder)

    def unicode(
        self, *, invert_color: bool = False, borders: bool = False, empty_square: str = "⭘", orientation: Color = WHITE
    ) -> str:
        """
        Returns a string representation of the board with Unicode pieces.
        Useful for pretty-printing to a terminal.

        :param invert_color: Invert color of the Unicode pieces.
        :param borders: Show borders and a coordinate margin.
        """
        builder = []
        for rank_index in range(7, -1, -1) if orientation else range(8):
            if borders:
                builder.append("  ")
                builder.append("-" * 17)
                builder.append("\n")

                builder.append(RANK_NAMES[rank_index])
                builder.append(" ")

            for i, file_index in enumerate(range(8) if orientation else range(7, -1, -1)):
                square_index = square(file_index, rank_index)

                if borders:
                    builder.append("|")
                elif i > 0:
                    builder.append(" ")

                piece = self.piece_at(square_index)

                if piece:
                    builder.append(piece.unicode_symbol(invert_color=invert_color))
                else:
                    builder.append(empty_square)

            if borders:
                builder.append("|")

            if borders or (rank_index > 0 if orientation else rank_index < 7):
                builder.append("\n")

        if borders:
            builder.append("  ")
            builder.append("-" * 17)
            builder.append("\n")
            letters = "a b c d e f g h" if orientation else "h g f e d c b a"
            builder.append("   " + letters)

        return "".join(builder)


    def copy(self: BaseBoardT) -> BaseBoardT:
        """Creates a copy of the board."""
        board = type(self)(None)

        board.pawns = self.pawns
        board.white_pawns = self.white_pawns
        board.black_pawns = self.black_pawns
        board.knights = self.knights
        board.white_knights = self.white_knights
        board.black_knights = self.black_knights
        board.bishops = self.bishops
        board.white_bishops = self.white_bishops
        board.black_bishops = self.black_bishops
        board.rooks = self.rooks
        board.white_rooks = self.white_rooks
        board.black_rooks = self.black_rooks
        board.queens = self.queens
        board.white_queens = self.white_queens
        board.black_queens = self.black_queens
        board.kings = self.kings
        board.white_kings = self.white_kings
        board.black_kings = self.black_kings

        board.occupied_co[WHITE] = self.occupied_co[WHITE]
        board.occupied_co[BLACK] = self.occupied_co[BLACK]
        board.occupied = self.occupied
        board.promoted = self.promoted

        return board

    def __copy__(self: BaseBoardT) -> BaseBoardT:
        return self.copy()

    def __deepcopy__(self: BaseBoardT, memo: Dict[int, object]) -> BaseBoardT:
        board = self.copy()
        memo[id(self)] = board
        return board

BoardT = TypeVar("BoardT", bound="Board")


class Board(BaseBoard):
    """
    A :class:`~chess.BaseBoard`, additional information representing
    a chess position

    Provides :data:`move generation <chess.Board.legal_moves>`, validation,
    :func:`parsing <chess.Board.parse_san()>`, attack generation,
    :func:`game end detection <chess.Board.is_game_over()>`,
    and the capability to :func:`make <chess.Board.push()>` and
    :func:`unmake <chess.Board.pop()>` moves.

    The board is initialized to the standard chess starting position,
    unless otherwise specified in the optional *fen* argument.
    If *fen* is ``None``, an empty board is created.

    Optionally supports *chess960*. In Chess960, castling moves are encoded
    by a king move to the corresponding rook square.
    Use :func:`chess.Board.from_chess960_pos()` to create a board with one
    of the Chess960 starting positions.

    It's safe to set :data:`~Board.turn`, :data:`~Board.castling_rights`,
    :data:`~Board.ep_square`, :data:`~Board.halfmove_clock` and
    :data:`~Board.fullmove_number` directly.

    .. warning::
        It is possible to set up and work with invalid positions. In this
        case, :class:`~chess.Board` implements a kind of "pseudo-chess"
        (useful to gracefully handle errors or to implement chess variants).
        Use :func:`~chess.Board.is_valid()` to detect invalid positions.
    """

    uci_variant: ClassVar[Optional[str]] = "chess"
    starting_fen: ClassVar[str] = STARTING_FEN

    turn: Color
    """The side to move (``chess.WHITE`` or ``chess.BLACK``)."""

    castling_rights: Bitboard
    """
    Bitmask of the rooks with castling rights.

    To test for specific squares:

    >>> import chess
    >>>
    >>> board = chess.Board()
    >>> bool(board.castling_rights & chess.BB_H1)  # White can castle with the h1 rook
    True

    To add a specific square:

    >>> board.castling_rights |= chess.BB_A1

    Use :func:`~chess.Board.set_castling_fen()` to set multiple castling
    rights. Also see :func:`~chess.Board.has_castling_rights()`,
    :func:`~chess.Board.has_kingside_castling_rights()`,
    :func:`~chess.Board.has_queenside_castling_rights()`,
    :func:`~chess.Board.clean_castling_rights()`.
    """

    ep_square: Optional[Square]
    """
    The potential en passant square on the third or sixth rank or ``None``.

    Use :func:`~chess.Board.has_legal_en_passant()` to test if en passant
    capturing would actually be possible on the next move.
    """

    fullmove_number: int
    """
    Counts move pairs. Starts at `1` and is incremented after every move
    of the black side.
    """

    halfmove_clock: int
    """The number of half-moves since the last capture or pawn move."""

    promoted: Bitboard
    """A bitmask of pieces that have been promoted."""

    def __init__(self: BoardT, fen: str|None = None, *args) -> None:
        BaseBoard.__init__(self, None)

        self.ep_square = None
        if fen:
            self.set_fen(fen)

    @property
    def legal_moves(self) -> LegalMoveGenerator:
        """
        A dynamic list of legal moves.

        >>> import chess
        >>>
        >>> board = chess.Board()
        >>> board.legal_moves.count()
        20
        >>> bool(board.legal_moves)
        True
        >>> move = chess.Move.from_uci("g1f3")
        >>> move in board.legal_moves
        True

        Wraps :func:`~chess.Board.generate_legal_moves()` and
        :func:`~chess.Board.is_legal()`.
        """
        return LegalMoveGenerator(self)

    def generate_pseudo_legal_moves(self, from_mask: Bitboard = BB_ALL, to_mask: Bitboard = BB_ALL) -> Iterator[tuple]:
        our_pieces = self.occupied_co[self.turn]

        pawns = (self.white_pawns if self.turn == WHITE else self.black_pawns) & from_mask

        if pawns:
            # Generate pawn captures.
            capturers = pawns
            for from_square in scan_reversed(capturers):
                for pieces, figure in (((self.black_pawns if self.turn == WHITE else self.white_pawns), PAWN),
                                       ((self.black_knights if self.turn == WHITE else self.white_knights), KNIGHT),
                                       ((self.black_bishops if self.turn == WHITE else self.white_bishops), BISHOP),
                                       ((self.black_rooks if self.turn == WHITE else self.white_rooks), ROOK),
                                       ((self.black_queens if self.turn == WHITE else self.white_queens), QUEEN),
                                       ((self.black_kings if self.turn == WHITE else self.white_kings), KING)):
                    targets = BB_PAWN_ATTACKS[self.turn][from_square] & pieces & to_mask

                    for to_square in scan_reversed(targets):
                        if square_rank(to_square) in [0, 7]:
                            yield (from_square, to_square, QUEEN, None, PAWN, figure, (figure-PAWN) if figure else QUEEN)
                            yield (from_square, to_square, ROOK, None, PAWN, figure, (figure-PAWN) if figure else ROOK)
                            yield (from_square, to_square, BISHOP, None, PAWN, figure, (figure-PAWN) if figure else BISHOP)
                            yield (from_square, to_square, KNIGHT, None, PAWN, figure, (figure-PAWN) if figure else KNIGHT)
                        else:
                            yield (from_square, to_square, None, None, PAWN, figure, (figure-PAWN) if figure else 0)

        non_pawns = our_pieces & ~self.pawns & from_mask
        for from_square in scan_reversed(non_pawns):
            moves, figure = self.attacks_mask(from_square)
            moves &= ~our_pieces & to_mask

            for to_square in scan_reversed(moves & (self.black_pawns if self.turn == WHITE else self.white_pawns)):
                yield (from_square, to_square, None, None, figure, PAWN, PAWN-figure)
            for to_square in scan_reversed(moves & (self.black_knights if self.turn == WHITE else self.white_knights)):
                yield (from_square, to_square, None, None, figure, KNIGHT, KNIGHT-figure)
            for to_square in scan_reversed(moves & (self.black_bishops if self.turn == WHITE else self.white_bishops)):
                yield (from_square, to_square, None, None, figure, BISHOP, BISHOP-figure)
            for to_square in scan_reversed(moves & (self.black_rooks if self.turn == WHITE else self.white_rooks)):
                yield (from_square, to_square, None, None, figure, ROOK, ROOK-figure)
            for to_square in scan_reversed(moves & (self.black_queens if self.turn == WHITE else self.white_queens)):
                yield (from_square, to_square, None, None, figure, QUEEN, QUEEN-figure)
            # for to_square in scan_reversed(moves & (self.black_kings if self.turn == WHITE else self.white_kings)):
            #     yield (from_square, to_square, None, None, figure, KING, KING-figure)
            for to_square in scan_reversed(moves & ~self.occupied_co[not self.turn]):
                yield (from_square, to_square, None, None, figure, None, 0)

        # Generate castling moves.
        if from_mask & self.kings:
            yield from self.generate_castling_moves(from_mask, to_mask)

        if not pawns:
            return

        # Generate en passant captures.
        if self.ep_square:
            yield from self.generate_pseudo_legal_ep(from_mask, to_mask)

        if self.turn == WHITE:
            single_moves = pawns << 8 & ~self.occupied
        else:
            single_moves = pawns >> 8 & ~self.occupied

        single_moves &= to_mask

        # Generate single pawn moves.
        for to_square in scan_reversed(single_moves):
            from_square = to_square + (8 if self.turn == BLACK else -8)

            if square_rank(to_square) in [0, 7]:
                yield (from_square, to_square, QUEEN, None, PAWN, None, QUEEN)
                yield (from_square, to_square, ROOK, None, PAWN, None, ROOK)
                yield (from_square, to_square, BISHOP, None, PAWN, None, BISHOP)
                yield (from_square, to_square, KNIGHT, None, PAWN, None, KNIGHT)
            else:
                yield (from_square, to_square, None, None, PAWN, None, 0)

        if self.turn == WHITE:
            double_moves = single_moves << 8 & ~self.occupied & (BB_RANK_3 | BB_RANK_4)
        else:
            double_moves = single_moves >> 8 & ~self.occupied & (BB_RANK_6 | BB_RANK_5)

        double_moves &= to_mask

        # Generate double pawn moves.
        for to_square in scan_reversed(double_moves):
            from_square = to_square + (16 if self.turn == BLACK else -16)
            yield (from_square, to_square, None, None, PAWN, None, 0)

    def generate_pseudo_legal_ep(self, from_mask: Bitboard = BB_ALL, to_mask: Bitboard = BB_ALL) -> Iterator[tuple]:
        if not self.ep_square or not BB_SQUARES[self.ep_square] & to_mask:
            return None

        if BB_SQUARES[self.ep_square] & self.occupied:
            return None

        capturers = (
            self.pawns
            & self.occupied_co[self.turn]
            & from_mask
            & BB_PAWN_ATTACKS[not self.turn][self.ep_square]
            & BB_RANKS[4 if self.turn else 3]
        )

        for capturer in scan_reversed(capturers):
            yield (capturer, self.ep_square, None, None, PAWN, PAWN, 0)

    def checkers_mask(self) -> Bitboard:
        king = self.king(self.turn)
        return BB_EMPTY if king is None else self.attackers_mask(not self.turn, king)

    def is_check(self) -> bool:
        """Tests if the current side to move is in check."""
        return bool(self.checkers_mask())

    def is_into_check(self, move: tuple) -> bool:
        king = self.king(self.turn)
        if king is None:
            return False

        # If already in check, look if it is an evasion.
        checkers = self.attackers_mask(not self.turn, king)
        if checkers and move not in self._generate_evasions(
            king, checkers, BB_SQUARES[move[FROM_SQUARE]], BB_SQUARES[move[TO_SQUARE]]
        ):
            return True

        return not self._is_safe(king, self._slider_blockers(king), move)

    def is_checkmate(self) -> bool:
        """Checks if the current position is a checkmate."""
        if not self.is_check():
            return False

        return not any(self.generate_legal_moves())

    def is_stalemate(self) -> bool:
        """Checks if the current position is a stalemate."""
        if self.is_check():
            return False

        return not any(self.generate_legal_moves())

    def is_insufficient_material(self) -> bool:
        """
        Checks if neither side has sufficient winning material
        (:func:`~chess.Board.has_insufficient_material()`).
        """
        return all(self.has_insufficient_material(color) for color in COLORS)

    def has_insufficient_material(self, color: Color) -> bool:
        """
        Checks if *color* has insufficient winning material.

        This is guaranteed to return ``False`` if *color* can still win the
        game.

        The converse does not necessarily hold:
        The implementation only looks at the material, including the colors
        of bishops, but not considering piece positions. So fortress
        positions or positions with forced lines may return ``False``, even
        though there is no possible winning line.
        """
        if self.occupied_co[color] & (self.pawns | self.rooks | self.queens):
            return False

        # Knights are only insufficient material if:
        # (1) We do not have any other pieces, including more than one knight.
        # (2) The opponent does not have pawns, knights, bishops or rooks.
        #     These would allow selfmate.
        if self.occupied_co[color] & self.knights:
            return popcount(self.occupied_co[color]) <= 2 and not (
                self.occupied_co[not color] & ~self.kings & ~self.queens
            )

        # Bishops are only insufficient material if:
        # (1) We do not have any other pieces, including bishops of the
        #     opposite color.
        # (2) The opponent does not have bishops of the opposite color,
        #     pawns or knights. These would allow selfmate.
        if self.occupied_co[color] & self.bishops:
            same_color = (not self.bishops & BB_DARK_SQUARES) or (not self.bishops & BB_LIGHT_SQUARES)
            return same_color and not self.pawns and not self.knights

        return True

    def push(self: BoardT, move: tuple, p=False) -> None:
        """
        Updates the position with the given *move* and puts it onto the
        move stack.

        >>> import chess
        >>>
        >>> board = chess.Board()
        >>>
        >>> Nf3 = chess.Move.from_uci("g1f3")
        >>> board.push(Nf3)  # Make the move

        >>> board.pop()  # Unmake the last move
        Move.from_uci('g1f3')

        Null moves just increment the move counters, switch turns and forfeit
        en passant capturing.

        .. warning::
            Moves are not checked for legality. It is the caller's
            responsibility to ensure that the move is at least pseudo-legal or
            a null move.
        """
        # Push move and remember board state.
        move = self._to_chess960(move)
        self.castling_rights = self.clean_castling_rights()  # Before pushing stack

        # Reset en passant square.
        ep_square = self.ep_square
        self.ep_square = None

        # Increment move counters.
        self.halfmove_clock += 1
        if self.turn == BLACK:
            self.fullmove_number += 1

        captured_piece_type = move[5]
        # Drops.
        if move[3]:
            self._set_piece_at(move[TO_SQUARE], move[3], self.turn, removed_piece_type=captured_piece_type)
            self.turn = not self.turn
            return

        # Zero the half-move clock.
        if self.is_zeroing(move):
            self.halfmove_clock = 0

        from_bb = BB_SQUARES[move[FROM_SQUARE]]
        to_bb = BB_SQUARES[move[TO_SQUARE]]

        promoted = bool(self.promoted & from_bb)
        piece_type = move[4]
        self._remove_piece_at(move[FROM_SQUARE], print=p, piece_type=piece_type)
        assert piece_type is not None, f"push() expects move to be pseudo-legal, but got {move} in {self.board_fen()}"

        # Update castling rights.
        self.castling_rights &= ~to_bb & ~from_bb
        if piece_type == KING and not promoted:
            if self.turn == WHITE:
                self.castling_rights &= ~BB_RANK_1
            else:
                self.castling_rights &= ~BB_RANK_8
        elif captured_piece_type == KING and not self.promoted & to_bb:
            print("tttttttt", move)
            if self.turn == WHITE and square_rank(move[TO_SQUARE]) == 7:
                self.castling_rights &= ~BB_RANK_8
            elif self.turn == BLACK and square_rank(move[TO_SQUARE]) == 0:
                self.castling_rights &= ~BB_RANK_1

        # Handle special pawn moves.
        if piece_type == PAWN:
            diff = move[TO_SQUARE] - move[FROM_SQUARE]

            if diff == 16 and square_rank(move[FROM_SQUARE]) == 1:
                self.ep_square = move[FROM_SQUARE] + 8
            elif diff == -16 and square_rank(move[FROM_SQUARE]) == 6:
                self.ep_square = move[FROM_SQUARE] - 8
            elif move[TO_SQUARE] == ep_square and abs(diff) in [7, 9] and not captured_piece_type:
                # Remove pawns captured en passant.
                down = -8 if self.turn == WHITE else 8
                capture_square = ep_square + down
                captured_piece_type = self._remove_piece_at(capture_square, captured_piece_type)

        # Promotion.
        if move[2]:
            promoted = True
            piece_type = move[2]

        # Castling.
        castling = piece_type == KING and self.occupied_co[self.turn] & to_bb
        if p:
            print(castling)
        if castling:
            a_side = square_file(move[TO_SQUARE]) < square_file(move[FROM_SQUARE])

            self._remove_piece_at(move[FROM_SQUARE])
            self._remove_piece_at(move[TO_SQUARE])

            if a_side:
                self._set_piece_at(C1 if self.turn == WHITE else C8, KING, self.turn)
                self._set_piece_at(D1 if self.turn == WHITE else D8, ROOK, self.turn)
                if p:
                        print("xd")
            else:
                self._set_piece_at(G1 if self.turn == WHITE else G8, KING, self.turn)
                self._set_piece_at(F1 if self.turn == WHITE else F8, ROOK, self.turn)
                if p:
                        print("jjj")

        # Put the piece on the target square.
        if not castling:
            self._set_piece_at(move[TO_SQUARE], piece_type, self.turn, promoted, captured_piece_type)

        self.turn = not self.turn

    def castling_shredder_fen(self) -> str:
        castling_rights = self.clean_castling_rights()
        if not castling_rights:
            return "-"

        builder = []

        for square in scan_reversed(castling_rights & BB_RANK_1):
            builder.append(FILE_NAMES[square_file(square)].upper())

        for square in scan_reversed(castling_rights & BB_RANK_8):
            builder.append(FILE_NAMES[square_file(square)])

        return "".join(builder)

    def castling_xfen(self) -> str:
        builder = []

        for color in COLORS:
            king = self.king(color)
            if king is None:
                continue

            king_file = square_file(king)
            backrank = BB_RANK_1 if color == WHITE else BB_RANK_8

            for rook_square in scan_reversed(self.clean_castling_rights() & backrank):
                rook_file = square_file(rook_square)
                a_side = rook_file < king_file

                other_rooks = self.occupied_co[color] & self.rooks & backrank & ~BB_SQUARES[rook_square]

                if any((square_file(other) < rook_file) == a_side for other in scan_reversed(other_rooks)):
                    ch = FILE_NAMES[rook_file]
                else:
                    ch = "q" if a_side else "k"

                builder.append(ch.upper() if color == WHITE else ch)

        if builder:
            return "".join(builder)
        else:
            return "-"

    def has_pseudo_legal_en_passant(self) -> bool:
        """Checks if there is a pseudo-legal en passant capture."""
        return self.ep_square is not None and any(self.generate_pseudo_legal_ep()[0])

    def has_legal_en_passant(self) -> bool:
        """Checks if there is a legal en passant capture."""
        return self.ep_square is not None and any(self.generate_legal_ep())

    def fen(
        self, *, shredder: bool = False, en_passant: _EnPassantSpec = "legal", promoted: Optional[bool] = None
    ) -> str:
        """
        Gets a FEN representation of the position.

        A FEN string (e.g.,
        ``rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1``) consists
        of the board part :func:`~chess.Board.board_fen()`, the
        :data:`~chess.Board.turn`, the castling part
        (:data:`~chess.Board.castling_rights`),
        the en passant square (:data:`~chess.Board.ep_square`),
        the :data:`~chess.Board.halfmove_clock`
        and the :data:`~chess.Board.fullmove_number`.

        :param shredder: Use :func:`~chess.Board.castling_shredder_fen()`
            and encode castling rights by the file of the rook
            (like ``HAha``) instead of the default
            :func:`~chess.Board.castling_xfen()` (like ``KQkq``).
        :param en_passant: By default, only fully legal en passant squares
            are included (:func:`~chess.Board.has_legal_en_passant()`).
            Pass ``fen`` to strictly follow the FEN specification
            (always include the en passant square after a two-step pawn move)
            or ``xfen`` to follow the X-FEN specification
            (:func:`~chess.Board.has_pseudo_legal_en_passant()`).
        :param promoted: Mark promoted pieces like ``Q~``. By default, this is
            only enabled in chess variants where this is relevant.
        """
        return " ".join(
            [
                self.epd(shredder=shredder, en_passant=en_passant, promoted=promoted),
                str(self.halfmove_clock),
                str(self.fullmove_number),
            ]
        )

    def set_fen(self, fen: str) -> None:
        """
        Parses a FEN and sets the position from it.

        :raises: :exc:`ValueError` if syntactically invalid. Use
            :func:`~chess.Board.is_valid()` to detect invalid positions.
        """
        parts = fen.split()

        # Board part.
        try:
            board_part = parts.pop(0)
        except IndexError:
            raise ValueError("empty fen")

        # Turn.
        try:
            turn_part = parts.pop(0)
        except IndexError:
            turn = WHITE
        else:
            if turn_part == "w":
                turn = WHITE
            elif turn_part == "b":
                turn = BLACK
            else:
                raise ValueError(f"expected 'w' or 'b' for turn part of fen: {fen!r}")

        # Validate castling part.
        try:
            castling_part = parts.pop(0)
        except IndexError:
            castling_part = "-"
        else:
            if not FEN_CASTLING_REGEX.match(castling_part):
                raise ValueError(f"invalid castling part in fen: {fen!r}")

        # En passant square.
        try:
            ep_part = parts.pop(0)
        except IndexError:
            ep_square = None
        else:
            try:
                ep_square = None if ep_part == "-" else SQUARE_NAMES.index(ep_part)
            except ValueError:
                raise ValueError(f"invalid en passant part in fen: {fen!r}")

        # Check that the half-move part is valid.
        try:
            halfmove_part = parts.pop(0)
        except IndexError:
            halfmove_clock = 0
        else:
            try:
                halfmove_clock = int(halfmove_part)
            except ValueError:
                raise ValueError(f"invalid half-move clock in fen: {fen!r}")

            if halfmove_clock < 0:
                raise ValueError(f"half-move clock cannot be negative: {fen!r}")

        # Check that the full-move number part is valid.
        # 0 is allowed for compatibility, but later replaced with 1.
        try:
            fullmove_part = parts.pop(0)
        except IndexError:
            fullmove_number = 1
        else:
            try:
                fullmove_number = int(fullmove_part)
            except ValueError:
                raise ValueError(f"invalid fullmove number in fen: {fen!r}")

            if fullmove_number < 0:
                raise ValueError(f"fullmove number cannot be negative: {fen!r}")

            fullmove_number = max(fullmove_number, 1)

        # All parts should be consumed now.
        if parts:
            raise ValueError(f"fen string has more parts than expected: {fen!r}")

        # Validate the board part and set it.
        self._set_board_fen(board_part)

        # Apply.
        self.turn = turn
        self._set_castling_fen(castling_part)
        self.ep_square = ep_square
        self.halfmove_clock = halfmove_clock
        self.fullmove_number = fullmove_number

    def _set_castling_fen(self, castling_fen: str) -> None:
        if not castling_fen or castling_fen == "-":
            self.castling_rights = BB_EMPTY
            return

        if not FEN_CASTLING_REGEX.match(castling_fen):
            raise ValueError(f"invalid castling fen: {castling_fen!r}")

        self.castling_rights = BB_EMPTY

        for flag in castling_fen:
            color = WHITE if flag.isupper() else BLACK
            flag = flag.lower()
            backrank = BB_RANK_1 if color == WHITE else BB_RANK_8
            rooks = self.occupied_co[color] & self.rooks & backrank
            king = self.king(color)

            if flag == "q":
                # Select the leftmost rook.
                if king is not None and lsb(rooks) < king:
                    self.castling_rights |= rooks & -rooks
                else:
                    self.castling_rights |= BB_FILE_A & backrank
            elif flag == "k":
                # Select the rightmost rook.
                rook = msb(rooks)
                if king is not None and king < rook:
                    self.castling_rights |= BB_SQUARES[rook]
                else:
                    self.castling_rights |= BB_FILE_H & backrank
            else:
                self.castling_rights |= BB_FILES[FILE_NAMES.index(flag)] & backrank

    def _epd_operations(self, operations: Mapping[str, Union[None, str, int, float, Move, Iterable[Move]]]) -> str:
        epd = []
        first_op = True

        for opcode, operand in operations.items():
            assert opcode != "-", "dash (-) is not a valid epd opcode"
            for blacklisted in [" ", "\n", "\t", "\r"]:
                assert blacklisted not in opcode, f"invalid character {blacklisted!r} in epd opcode: {opcode!r}"

            if not first_op:
                epd.append(" ")
            first_op = False
            epd.append(opcode)

            if operand is None:
                epd.append(";")
            elif isinstance(operand, Move):
                epd.append(" ")
                epd.append(self.san(operand))
                epd.append(";")
            elif isinstance(operand, int):
                epd.append(f" {operand};")
            elif isinstance(operand, float):
                assert math.isfinite(operand), f"expected numeric epd operand to be finite, got: {operand}"
                epd.append(f" {operand};")
            elif opcode == "pv" and not isinstance(operand, str) and hasattr(operand, "__iter__"):
                position = self.copy()
                for move in operand:
                    epd.append(" ")
                    epd.append(position.san_and_push(move))
                epd.append(";")
            elif opcode in ["am", "bm"] and not isinstance(operand, str) and hasattr(operand, "__iter__"):
                for san in sorted(self.san(move) for move in operand):
                    epd.append(" ")
                    epd.append(san)
                epd.append(";")
            else:
                # Append as escaped string.
                epd.append(' "')
                epd.append(
                    str(operand)
                    .replace("\\", "\\\\")
                    .replace("\t", "\\t")
                    .replace("\r", "\\r")
                    .replace("\n", "\\n")
                    .replace('"', '\\"')
                )
                epd.append('";')

        return "".join(epd)

    def epd(
        self,
        *,
        shredder: bool = False,
        en_passant: _EnPassantSpec = "legal",
        promoted: Optional[bool] = None,
        **operations: Union[None, str, int, float, Move, Iterable[Move]],
    ) -> str:
        """
        Gets an EPD representation of the current position.

        See :func:`~chess.Board.fen()` for FEN formatting options (*shredder*,
        *ep_square* and *promoted*).

        EPD operations can be given as keyword arguments. Supported operands
        are strings, integers, finite floats, legal moves and ``None``.
        Additionally, the operation ``pv`` accepts a legal variation as
        a list of moves. The operations ``am`` and ``bm`` accept a list of
        legal moves in the current position.

        The name of the field cannot be a lone dash and cannot contain spaces,
        newlines, carriage returns or tabs.

        *hmvc* and *fmvn* are not included by default. You can use:

        >>> import chess
        >>>
        >>> board = chess.Board()
        >>> board.epd(hmvc=board.halfmove_clock, fmvn=board.fullmove_number)
        'rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - hmvc 0; fmvn 1;'
        """
        if en_passant == "fen":
            ep_square = self.ep_square
        elif en_passant == "xfen":
            ep_square = self.ep_square if self.has_pseudo_legal_en_passant() else None
        else:
            ep_square = self.ep_square if self.has_legal_en_passant() else None

        epd = [
            self.board_fen(promoted=promoted),
            "w" if self.turn == WHITE else "b",
            self.castling_shredder_fen() if shredder else self.castling_xfen(),
            SQUARE_NAMES[ep_square] if ep_square is not None else "-",
        ]

        if operations:
            epd.append(self._epd_operations(operations))

        return " ".join(epd)

    def san(self, move: Move) -> str:
        """
        Gets the standard algebraic notation of the given move in the context
        of the current position.
        """
        return self._algebraic(move)

    def san_and_push(self, move: Move) -> str:
        return self._algebraic_and_push(move)

    def _algebraic(self, move: Move, *, long: bool = False) -> str:
        san = self._algebraic_and_push(move, long=long)
        self.pop()
        return san

    def _algebraic_and_push(self, move: Move, *, long: bool = False) -> str:
        san = self._algebraic_without_suffix(move, long=long)

        # Look ahead for check or checkmate.
        self.push(move)
        is_check = self.is_check()
        is_checkmate = (is_check and self.is_checkmate())

        # Add check or checkmate suffix.
        if is_checkmate and move:
            return san + "#"
        elif is_check and move:
            return san + "+"
        else:
            return san

    def _algebraic_without_suffix(self, move: Move, *, long: bool = False) -> str:
        # Null move.
        if not move:
            return "--"

        # Drops.
        if move.drop:
            san = ""
            if move.drop != PAWN:
                san = piece_symbol(move.drop).upper()
            san += "@" + SQUARE_NAMES[move.to_square]
            return san

        # Castling.
        if self.is_castling(move):
            if square_file(move.to_square) < square_file(move.from_square):
                return "O-O-O"
            else:
                return "O-O"

        piece_type = self.piece_type_at(move.from_square)
        assert piece_type, f"san() and lan() expect move to be legal or null, but got {move} in {self.fen()}"
        capture = self.is_capture(move)

        if piece_type == PAWN:
            san = ""
        else:
            san = piece_symbol(piece_type).upper()

        if long:
            san += SQUARE_NAMES[move.from_square]
        elif piece_type != PAWN:
            # Get ambiguous move candidates.
            # Relevant candidates: not exactly the current move,
            # but to the same square.
            others = 0
            from_mask = self.pieces_mask(piece_type, self.turn)
            from_mask &= ~BB_SQUARES[move.from_square]
            to_mask = BB_SQUARES[move.to_square]
            for candidate in self.generate_legal_moves(from_mask, to_mask):
                others |= BB_SQUARES[candidate.from_square]

            # Disambiguate.
            if others:
                row, column = False, False

                if others & BB_RANKS[square_rank(move.from_square)]:
                    column = True

                if others & BB_FILES[square_file(move.from_square)]:
                    row = True
                else:
                    column = True

                if column:
                    san += FILE_NAMES[square_file(move.from_square)]
                if row:
                    san += RANK_NAMES[square_rank(move.from_square)]
        elif capture:
            san += FILE_NAMES[square_file(move.from_square)]

        # Captures.
        if capture:
            san += "x"
        elif long:
            san += "-"

        # Destination square.
        san += SQUARE_NAMES[move.to_square]

        # Promotion.
        if move.promotion:
            san += "=" + piece_symbol(move.promotion).upper()

        return san

    def is_en_passant(self, move: tuple) -> bool:
        """Checks if the given pseudo-legal move is an en passant capture."""
        return (
            self.ep_square == move[TO_SQUARE]
            and bool(self.pawns & BB_SQUARES[move[FROM_SQUARE]])
            and abs(move[TO_SQUARE] - move[FROM_SQUARE]) in [7, 9]
            and not self.occupied & BB_SQUARES[move[FROM_SQUARE]]
        )

    def is_capture(self, move: Move) -> bool:
        """Checks if the given pseudo-legal move is a capture."""
        touched = BB_SQUARES[move.from_square] ^ BB_SQUARES[move.to_square]
        return bool(touched & self.occupied_co[not self.turn]) or self.is_en_passant(move)

    def is_zeroing(self, move: tuple) -> bool:
        """Checks if the given pseudo-legal move is a capture or pawn move."""
        touched = BB_SQUARES[move[FROM_SQUARE]] ^ BB_SQUARES[move[TO_SQUARE]]
        return bool(touched & self.pawns or touched & self.occupied_co[not self.turn] or move[3] == PAWN)

    def is_castling(self, move: tuple) -> bool:
        """Checks if the given pseudo-legal move is a castling move."""
        if self.kings & BB_SQUARES[move[FROM_SQUARE]]:
            diff = square_file(move[FROM_SQUARE]) - square_file(move[TO_SQUARE])
            return abs(diff) > 1 or bool((self.white_rooks if self.turn == WHITE else self.black_rooks) & BB_SQUARES[move[TO_SQUARE]])
        return False

    def clean_castling_rights(self) -> Bitboard:
        """
        Returns valid castling rights filtered from
        :data:`~chess.Board.castling_rights`.
        """

        castling = self.castling_rights & self.rooks
        white_castling = castling & BB_RANK_1 & self.occupied_co[WHITE]
        black_castling = castling & BB_RANK_8 & self.occupied_co[BLACK]

        # The rooks must be on a1, h1, a8 or h8.
        white_castling &= BB_A1 | BB_H1
        black_castling &= BB_A8 | BB_H8

        # The kings must be on e1 or e8.
        if not self.white_kings & ~self.promoted & BB_E1:
            white_castling = 0
        if not self.black_kings & ~self.promoted & BB_E8:
            black_castling = 0

        return white_castling | black_castling

    def has_kingside_castling_rights(self, color: Color) -> bool:
        """
        Checks if the given side has kingside (that is h-side in Chess960)
        castling rights.
        """
        backrank = BB_RANK_1 if color == WHITE else BB_RANK_8
        king_mask = (self.white_kings if color == WHITE else self.black_kings) & backrank & ~self.promoted
        if not king_mask:
            return False

        castling_rights = self.clean_castling_rights() & backrank
        while castling_rights:
            rook = castling_rights & -castling_rights

            if rook > king_mask:
                return True

            castling_rights &= castling_rights - 1

        return False

    def has_queenside_castling_rights(self, color: Color) -> bool:
        """
        Checks if the given side has queenside (that is a-side in Chess960)
        castling rights.
        """
        backrank = BB_RANK_1 if color == WHITE else BB_RANK_8
        king_mask = (self.white_kings if color == WHITE else self.black_kings) & backrank & ~self.promoted
        if not king_mask:
            return False

        castling_rights = self.clean_castling_rights() & backrank
        while castling_rights:
            rook = castling_rights & -castling_rights

            if rook < king_mask:
                return True

            castling_rights &= castling_rights - 1

        return False

    def _ep_skewered(self, king: Square, capturer: Square) -> bool:
        # Handle the special case where the king would be in check if the
        # pawn and its capturer disappear from the rank.

        # Vertical skewers of the captured pawn are not possible. (Pins on
        # the capturer are not handled here.)
        assert self.ep_square is not None

        last_double = self.ep_square + (-8 if self.turn == WHITE else 8)

        occupancy = self.occupied & ~BB_SQUARES[last_double] & ~BB_SQUARES[capturer] | BB_SQUARES[self.ep_square]

        # Horizontal attack on the fifth or fourth rank.
        horizontal_attackers = self.occupied_co[not self.turn] & (self.rooks | self.queens)
        if BB_RANK_ATTACKS[king][BB_RANK_MASKS[king] & occupancy] & horizontal_attackers:
            return True

        # Diagonal skewers. These are not actually possible in a real game,
        # because if the latest double pawn move covers a diagonal attack,
        # then the other side would have been in check already.
        diagonal_attackers = self.occupied_co[not self.turn] & (self.bishops | self.queens)
        if BB_DIAG_ATTACKS[king][BB_DIAG_MASKS[king] & occupancy] & diagonal_attackers:
            return True

        return False

    def _slider_blockers(self, king: Square) -> Bitboard:
        rooks_and_queens = self.rooks | self.queens
        bishops_and_queens = self.bishops | self.queens

        snipers = (
            (BB_RANK_ATTACKS[king][0] & rooks_and_queens)
            | (BB_FILE_ATTACKS[king][0] & rooks_and_queens)
            | (BB_DIAG_ATTACKS[king][0] & bishops_and_queens)
        )

        blockers = 0

        for sniper in scan_reversed(snipers & self.occupied_co[not self.turn]):
            b = between(king, sniper) & self.occupied

            # Add to blockers if exactly one piece in-between.
            if b and BB_SQUARES[msb(b)] == b:
                blockers |= b

        return blockers & self.occupied_co[self.turn]

    def _is_safe(self, king: Square, blockers: Bitboard, move: tuple) -> bool:
        if move[FROM_SQUARE] == king:
            if self.is_castling(move):
                return True
            else:
                return not self.is_attacked_by(not self.turn, move[TO_SQUARE])
        elif self.is_en_passant(move):
            return bool(
                self.pin_mask(self.turn, move[FROM_SQUARE]) & BB_SQUARES[move[TO_SQUARE]]
                and not self._ep_skewered(king, move[FROM_SQUARE])
            )
        else:
            return bool(
                not blockers & BB_SQUARES[move[FROM_SQUARE]]
                or ray(move[FROM_SQUARE], move[TO_SQUARE]) & BB_SQUARES[king]
            )

    def _generate_evasions(
        self, king: Square, checkers: Bitboard, from_mask: Bitboard = BB_ALL, to_mask: Bitboard = BB_ALL
    ) -> Iterator[tuple]:
        sliders = checkers & (self.bishops | self.rooks | self.queens)

        attacked = 0
        for checker in scan_reversed(sliders):
            attacked |= ray(king, checker) & ~BB_SQUARES[checker]

        checker = msb(checkers)
        if BB_SQUARES[checker] == checkers:
            # Capture or block a single checker.
            target = between(king, checker) | checkers

            yield from self.generate_pseudo_legal_moves(~self.kings & from_mask, target & to_mask)

            # Capture the checking pawn en passant
            if self.ep_square and not BB_SQUARES[self.ep_square] & target:
                last_double = self.ep_square + (-8 if self.turn == WHITE else 8)
                if last_double == checker:
                    yield from self.generate_pseudo_legal_ep(from_mask, to_mask)

        if BB_SQUARES[king] & from_mask:
            base = BB_KING_ATTACKS[king] & ~attacked & to_mask
            for to_square in scan_reversed(base & (self.black_queens if self.turn == WHITE else self.white_queens)):
                yield (king, to_square, None, None, KING, QUEEN, QUEEN)
            for to_square in scan_reversed(base & (self.black_rooks if self.turn == WHITE else self.white_rooks)):
                yield (king, to_square, None, None, KING, ROOK, ROOK)
            for to_square in scan_reversed(base & (self.black_bishops if self.turn == WHITE else self.white_bishops)):
                yield (king, to_square, None, None, KING, BISHOP, BISHOP)
            for to_square in scan_reversed(base & (self.black_knights if self.turn == WHITE else self.white_knights)):
                yield (king, to_square, None, None, KING, KNIGHT, KNIGHT)
            for to_square in scan_reversed(base & (self.black_pawns if self.turn == WHITE else self.white_pawns)):
                yield (king, to_square, None, None, KING, PAWN, PAWN)
            for to_square in scan_reversed(base & ~self.occupied):
                yield (king, to_square, None, None, KING, None, 0)


    def generate_legal_moves(self, from_mask: Bitboard = BB_ALL, to_mask: Bitboard = BB_ALL) -> Iterator[Move]:
        king = msb(self.white_kings if self.turn == WHITE else self.black_kings)
        blockers = self._slider_blockers(king)
        checkers = self.attackers_mask(not self.turn, king)
        if checkers:
            for move in sorted(self._generate_evasions(king, checkers, from_mask, to_mask), key=lambda move: move[6], reverse=True):
                if self._is_safe(king, blockers, move):
                    yield move
        else:
            for move in sorted(self.generate_pseudo_legal_moves(from_mask, to_mask), key=lambda move: move[6], reverse=True):
                if self._is_safe(king, blockers, move):
                    yield move

    def generate_legal_ep(self, from_mask: Bitboard = BB_ALL, to_mask: Bitboard = BB_ALL) -> Iterator[Move]:
        for move, figure in self.generate_pseudo_legal_ep(from_mask, to_mask):
            if move is None:
                continue
            if not self.is_into_check(move):
                yield move

    def _attacked_for_king(self, path: Bitboard, occupied: Bitboard) -> bool:
        return any(self._attackers_mask(not self.turn, sq, occupied) for sq in scan_reversed(path))

    def generate_castling_moves(self, from_mask: Bitboard = BB_ALL, to_mask: Bitboard = BB_ALL) -> Iterator[Move]:
        backrank = BB_RANK_1 if self.turn == WHITE else BB_RANK_8
        king = self.occupied_co[self.turn] & self.kings & ~self.promoted & backrank & from_mask
        king &= -king
        if not king:
            return

        bb_c = BB_FILE_C & backrank
        bb_d = BB_FILE_D & backrank
        bb_f = BB_FILE_F & backrank
        bb_g = BB_FILE_G & backrank

        for candidate in scan_reversed(self.clean_castling_rights() & backrank & to_mask):
            rook = BB_SQUARES[candidate]

            a_side = rook < king
            king_to = bb_c if a_side else bb_g
            rook_to = bb_d if a_side else bb_f

            king_path = between(msb(king), msb(king_to))
            rook_path = between(candidate, msb(rook_to))

            if not (
                (self.occupied ^ king ^ rook) & (king_path | rook_path | king_to | rook_to)
                or self._attacked_for_king(king_path | king, self.occupied ^ king)
                or self._attacked_for_king(king_to, self.occupied ^ king ^ rook ^ rook_to)
            ):
                yield self._from_chess960(msb(king), candidate)

    def _from_chess960(
        self,
        from_square: Square,
        to_square: Square,
        promotion: Optional[PieceType] = None,
        drop: Optional[PieceType] = None,
            piece: int|None = None
    ) -> tuple:
        if promotion is None and drop is None:
            if from_square == E1 and self.kings & BB_E1:
                if to_square == H1:
                    return (E1, G1, None, None, KING, None, 0)
                elif to_square == A1:
                    return (E1, C1, None, None, KING, None,0)
            elif from_square == E8 and self.kings & BB_E8:
                if to_square == H8:
                    return (E8, G8, None, None, KING, None,0)
                elif to_square == A8:
                    return (E8, C8, None, None, KING, None, 0)
        print("xxdxdxdxdxdxd")
        return (from_square, to_square, promotion, drop, 0)

    def _to_chess960(self, move: tuple) -> tuple:
        if move[FROM_SQUARE] == E1 and self.white_kings & BB_E1:
            if move[TO_SQUARE] == G1 and not self.white_rooks & BB_G1:
                return (E1, H1, None, None, KING, KING)
            elif move[TO_SQUARE] == C1 and not self.white_rooks & BB_C1:
                return (E1, A1, None, None, KING, KING)
        elif move[FROM_SQUARE] == E8 and self.black_kings & BB_E8:
            if move[TO_SQUARE] == G8 and not self.black_rooks & BB_G8:
                return (E8, H8, None, None, KING, KING)
            elif move[TO_SQUARE] == C8 and not self.black_rooks & BB_C8:
                return (E8, A8, None, None, KING, KING)

        return move

    def __repr__(self) -> str:
        return f"{type(self).__name__}({self.fen()!r})"

    def copy(self: BoardT, *args) -> BoardT:
        """
        Creates a copy of the board.

        Defaults to copying the entire move stack. Alternatively, *stack* can
        be ``False``, or an integer to copy a limited number of moves.
        """
        board = super().copy()

        board.ep_square = self.ep_square
        board.castling_rights = self.castling_rights
        board.turn = self.turn
        board.fullmove_number = self.fullmove_number
        board.halfmove_clock = self.halfmove_clock

        return board


IntoSquareSet = Union[SupportsInt, Iterable[Square]]
