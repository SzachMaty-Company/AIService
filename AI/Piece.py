class Piece:
    Empty = 0
    King = 1
    Pawn = 2
    Knight = 3
    Bishop = 5
    Rook = 6
    Queen = 7

    White = 8
    Black = 16

    typeMask = 0b00111
    blackMask = 0b10000
    whiteMask = 0b01000
    colourMask = whiteMask | blackMask

    @classmethod
    def IsColour(cls, piece, colour) -> bool:
        return (piece & cls.colourMask) == colour

    @classmethod
    def Colour(cls, piece) -> int:
        return piece & cls.colourMask

    @classmethod
    def PieceType(cls, piece) -> int:
        return piece & cls.typeMask

    @staticmethod
    def IsRookOrQueen(piece) -> bool:
        return (piece & 0b110) == 0b110

    @staticmethod
    def IsBishopOrQueen(piece) -> bool:
        return (piece & 0b101) == 0b101

    @staticmethod
    def IsSlidingPiece(piece) -> bool:
        return (piece & 0b100) != 0
