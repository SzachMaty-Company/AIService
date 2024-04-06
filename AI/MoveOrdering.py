from MoveGenerator import MoveGenerator
from Move import Move
from Board import Board
from Piece import Piece
from Evaluation import *
from BitBoardUtility import ContainsSquare
from TranspositionTable import TranspositionTable


class MoveOrdering:
    moveScores: list[int]
    maxMoveCount = 218

    squareControlledByOpponentPawnPenalty = 350
    capturedPieceValueMultiplier = 10

    moveGenerator: MoveGenerator
    transpositionTable: TranspositionTable
    invalidMove: Move

    def __init__(self, moveGenerator: MoveGenerator, tt: TranspositionTable):
        self.moveScores = [0] * self.maxMoveCount
        self.moveGenerator = moveGenerator
        self.transpositionTable = tt
        self.invalidMove = Move.InvalidMove()

    def OrderMoves(self, board: Board, moves: list[Move], useTT) -> list[Move]:
        hashMove: Move = self.invalidMove
        if useTT:
            hashMove = self.transpositionTable.GetStoredMove()

        for i in range(len(moves)):
            score = 0
            movePieceType = Piece.PieceType(board.Square[moves[i].StartSquare()])
            capturePieceType = Piece.PieceType(board.Square[moves[i].TargetSquare()])
            flag = moves[i].MoveFlag

            if capturePieceType != Piece.Empty:
                # Order moves to try capturing the most valuable opponent piece with least valuable of own pieces first
                # The capturedPieceValueMultiplier is used to make even 'bad' captures like QxP rank above non-captures
                score = self.capturedPieceValueMultiplier * self.GetPieceValue(capturePieceType) - self.GetPieceValue(
                    movePieceType
                )

            if movePieceType == Piece.Pawn:
                if flag == Move.Flag.PromoteToQueen:
                    score += queenValue
                elif flag == Move.Flag.PromoteToKnight:
                    score += knightValue
                elif flag == Move.Flag.PromoteToRook:
                    score += rookValue
                elif flag == Move.Flag.PromoteToBishop:
                    score += bishopValue
            else:
                # Penalize moving piece to a square attacked by opponent pawn
                if ContainsSquare(self.moveGenerator.opponentPawnAttackMap, moves[i].TargetSquare()):
                    score -= self.squareControlledByOpponentPawnPenalty

            if Move.SameMove(moves[i], hashMove):
                score += 10000

            self.moveScores[i] = score

        return self.Sort(moves)

    def GetPieceValue(self, pieceType) -> int:
        match (pieceType):
            case Piece.Queen:
                return queenValue
            case Piece.Rook:
                return rookValue
            case Piece.Knight:
                return knightValue
            case Piece.Bishop:
                return bishopValue
            case Piece.Pawn:
                return pawnValue
            case _:
                return 0

    def Sort(self, moves: list[Move]) -> list[Move]:
        # Sort the moves list based on scores
        for i in range(len(moves)):
            for j in range(i + 1, len(moves), -1):
                swapIndex = j - 1
                if self.moveScores[swapIndex] < self.moveScores[j]:
                    (moves[j], moves[swapIndex]) = (moves[swapIndex], moves[j])
                    (self.moveScores[j], self.moveScores[swapIndex]) = (self.moveScores[swapIndex], self.moveScores[j])

        return moves
