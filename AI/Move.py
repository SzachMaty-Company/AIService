from Piece import Piece
from BoardRepresentation import *


class Move:
    class Flag:
        Empty = 0
        EnPassantCapture = 1
        Castling = 2
        PromoteToQueen = 3
        PromoteToKnight = 4
        PromoteToRook = 5
        PromoteToBishop = 6
        PawnTwoForward = 7

    moveValue = None

    startSquareMask = 0b0000000000111111
    targetSquareMask = 0b0000111111000000
    flagMask = 0b1111000000000000
    x: int
    y: int
    z: int

    def __init__(self, startSquare, targetSquare, flag=0):
        if flag:
            self.moveValue = startSquare | targetSquare << 6 | flag << 12
        else:
            self.moveValue = startSquare | targetSquare << 6
        self.x = startSquare
        self.y = targetSquare
        self.z = flag
    def StartSquare(self) -> int:
        return self.moveValue & self.startSquareMask

    def TargetSquare(self) -> int:
        return (self.moveValue & self.targetSquareMask) >> 6

    def IsPromotion(self) -> bool:
        flag = self.MoveFlag
        return flag in [
            Move.Flag.PromoteToQueen,
            Move.Flag.PromoteToRook,
            Move.Flag.PromoteToKnight,
            Move.Flag.PromoteToBishop,
        ]

    def MoveFlag(self) -> int:
        return self.moveValue >> 12

    def PromotionPieceType(self) -> int:
        match self.MoveFlag:
            case Move.Flag.PromoteToRook:
                return Piece.Rook
            case Move.Flag.PromoteToKnight:
                return Piece.Knight
            case Move.Flag.PromoteToBishop:
                return Piece.Bishop
            case Move.Flag.PromoteToQueen:
                return Piece.Queen
            case _:
                return Piece.Empty

    @staticmethod
    def InvalidMove() -> "Move":
        instance = Move.__new__(Move)
        instance.moveValue = 0
        return instance

    @staticmethod
    def SameMove(a, b) -> bool:
        return a.moveValue == b.moveValue

    def Value(self) -> int:
        return self.moveValue

    def IsInvalid(self) -> bool:
        return self.moveValue == 0

    def Name(self) -> str:
        # TODO promojce
        return f"{SquareNameFromIndex(self.StartSquare())}{SquareNameFromIndex(self.TargetSquare())}"
