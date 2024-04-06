from Board import Board
from const import *
from PrecomputedMoveData import *
from PieceList import *
from PieceSquareTable import PieceSquareTable

pawnValue = 100
knightValue = 320
bishopValue = 330
rookValue = 500
queenValue = 900


class Evaluation:
    endgameMaterialStart = rookValue * 2 + bishopValue + knightValue
    board: Board

    # Performs static evaluation of the current position.
    # The position is assumed to be 'quiet', i.e no captures are available that could drastically affect the evaluation.
    # The score that's returned is given from the perspective of whoever's turn it is to move.
    # So a positive score means the player who's turn it is to move has an advantage, while a negative score indicates a disadvantage.
    def Evaluate(self, board: Board) -> int:
        self.board = board
        whiteEval = 0
        blackEval = 0

        whiteMaterial = self.CountMaterial(WhiteIndex)
        blackMaterial = self.CountMaterial(BlackIndex)

        whiteMaterialWithoutPawns = whiteMaterial - len(board.pawns[WhiteIndex]) * pawnValue
        blackMaterialWithoutPawns = blackMaterial - len(board.pawns[BlackIndex]) * pawnValue
        whiteEndgamePhaseWeight = self.EndgamePhaseWeight(whiteMaterialWithoutPawns)
        blackEndgamePhaseWeight = self.EndgamePhaseWeight(blackMaterialWithoutPawns)

        whiteEval += whiteMaterial
        blackEval += blackMaterial
        whiteEval += self.MopUpEval(WhiteIndex, BlackIndex, whiteMaterial, blackMaterial, blackEndgamePhaseWeight)
        blackEval += self.MopUpEval(BlackIndex, WhiteIndex, blackMaterial, whiteMaterial, whiteEndgamePhaseWeight)

        whiteEval += self.EvaluatePieceSquareTables(WhiteIndex, blackEndgamePhaseWeight)
        blackEval += self.EvaluatePieceSquareTables(BlackIndex, whiteEndgamePhaseWeight)

        eval = whiteEval - blackEval

        perspective = 1 if self.board.WhiteToMove else -1
        return eval * perspective

    def EndgamePhaseWeight(self, materialCountWithoutPawns) -> float:
        multiplier = 1 / self.endgameMaterialStart
        return 1 - min(1, materialCountWithoutPawns * multiplier)

    def MopUpEval(self, friendlyIndex, opponentIndex, myMaterial, opponentMaterial, endgameWeight) -> int:
        mopUpScore = 0
        if myMaterial > opponentMaterial + pawnValue * 2 and endgameWeight > 0:
            friendlyKingSquare = self.board.KingSquare[friendlyIndex]
            opponentKingSquare = self.board.KingSquare[opponentIndex]
            mopUpScore += centreManhattanDistance[opponentKingSquare] * 10
            # use ortho dst to promote direct opposition
            mopUpScore += (14 - NumRookMovesToReachSquare(friendlyKingSquare, opponentKingSquare)) * 4

            return int(mopUpScore * endgameWeight)
        return 0

    def CountMaterial(self, colourIndex) -> int:
        material = 0
        material += len(self.board.pawns[colourIndex]) * pawnValue
        material += len(self.board.knights[colourIndex]) * knightValue
        material += len(self.board.bishops[colourIndex]) * bishopValue
        material += len(self.board.rooks[colourIndex]) * rookValue
        material += len(self.board.queens[colourIndex]) * queenValue

        return material

    def EvaluatePieceSquareTables(self, colourIndex, endgamePhaseWeight) -> int:
        value = 0
        isWhite = colourIndex == WhiteIndex
        value += self.EvaluatePieceSquareTable(PieceSquareTable.pawns, self.board.pawns[colourIndex], isWhite)
        value += self.EvaluatePieceSquareTable(PieceSquareTable.rooks, self.board.rooks[colourIndex], isWhite)
        value += self.EvaluatePieceSquareTable(PieceSquareTable.knights, self.board.knights[colourIndex], isWhite)
        value += self.EvaluatePieceSquareTable(PieceSquareTable.bishops, self.board.bishops[colourIndex], isWhite)
        value += self.EvaluatePieceSquareTable(PieceSquareTable.queens, self.board.queens[colourIndex], isWhite)
        kingEarlyPhase = PieceSquareTable.Read(PieceSquareTable.kingMiddle, self.board.KingSquare[colourIndex], isWhite)
        value += int(kingEarlyPhase * (1 - endgamePhaseWeight))
        # value += PieceSquareTable.Read (PieceSquareTable.kingMiddle, board.KingSquare[colourIndex], isWhite)

        return value

    def EvaluatePieceSquareTable(self, table: list[int], pieceList: PieceList, isWhite) -> int:
        value = 0
        for i in range(len(pieceList)):
            value += PieceSquareTable.Read(table, pieceList[i], isWhite)

        return value
